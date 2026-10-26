// Writer-local truth extrapolation for the pinned ACTS build.
#pragma once

#include "Acts/EventData/BoundTrackParameters.hpp"
#include "Acts/MagneticField/MagneticFieldProvider.hpp"
#include "Acts/MagneticField/MagneticFieldError.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Propagator/SympyStepper.hpp"
#include "Acts/Surfaces/PerigeeSurface.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ActsExamples::detail {

struct TruthPerigeeResult {
  std::optional<Acts::BoundTrackParameters> parameters;
  double pathLength = 0.;
  std::string error;
  unsigned int retries = 0;
  unsigned int boundaries = 0;
};

struct DampedPerigeeReached : Acts::ForcedSurfaceReached {
  template <typename State, typename Stepper, typename Navigator>
  bool checkAbort(State& state, const Stepper& stepper,
                  const Navigator& navigator, const Acts::Logger& logger) const {
    bool reached = Acts::ForcedSurfaceReached::checkAbort(state, stepper, navigator, logger);
    if (!reached && surface != nullptr) {
      auto intersection = surface->intersect(
          state.geoContext, stepper.position(state.stepping),
          state.options.direction * stepper.direction(state.stepping),
          boundaryTolerance, state.options.surfaceTolerance).closest();
      if (intersection.isValid() && std::isfinite(intersection.pathLength())) {
        stepper.updateStepSize(state.stepping, .5 * intersection.pathLength(),
                               Acts::ConstrainedStep::Type::Actor);
      }
    }
    return reached;
  }
};

// Reject field queries outside the configured recovery domain. This wrapper is
// used only for retries; the legacy attempt always sees the unchanged field.
class RecoveryField final : public Acts::MagneticFieldProvider {
 public:
  RecoveryField(std::shared_ptr<const Acts::MagneticFieldProvider> field,
                double maxR, double maxAbsZ)
      : m_field(std::move(field)), m_maxR(maxR), m_maxAbsZ(maxAbsZ) {}

  bool contains(const Acts::Vector3& position) const {
    return position.allFinite() && position.head<2>().norm() < m_maxR &&
           std::abs(position.z()) < m_maxAbsZ;
  }

  Cache makeCache(const Acts::MagneticFieldContext& context) const override {
    return m_field->makeCache(context);
  }

  Acts::Result<Acts::Vector3> getField(const Acts::Vector3& position,
                                     Cache& cache) const override {
    if (!contains(position)) {
      return Acts::Result<Acts::Vector3>::failure(Acts::MagneticFieldError::OutOfBounds);
    }
    return m_field->getField(position, cache);
  }

 private:
  std::shared_ptr<const Acts::MagneticFieldProvider> m_field;
  double m_maxR;
  double m_maxAbsZ;
};

inline TruthPerigeeResult truthPerigeeAttempt(
    const Acts::GeometryContext& gctx, const Acts::MagneticFieldContext& mctx,
    const Acts::BoundTrackParameters& start, const Acts::Surface& target,
    std::shared_ptr<const Acts::MagneticFieldProvider> field,
    const Acts::Logger& logger, unsigned int mode) {
  const double tolerance = mode == 0 ? 1e-4 : 1e-6;
  auto intersection = target.intersect(gctx, start.position(gctx), start.direction(),
                                        Acts::BoundaryTolerance::Infinite(), tolerance).closest();
  if (mode != 0 && (!intersection.isValid() || !std::isfinite(intersection.pathLength()))) {
    return {std::nullopt, 0., "invalid initial target intersection"};
  }
  using Propagator = Acts::Propagator<Acts::SympyStepper>;
  Propagator propagator(Acts::SympyStepper(field), Propagator::Navigator{},
                        logger.cloneWithSuffix("PerigeePropagator"));
  Propagator::Options<> options(gctx, mctx);
  options.direction = Acts::Direction::fromScalarZeroAsPositive(intersection.pathLength());
  options.surfaceTolerance = tolerance;
  options.loopProtection = true;
  options.maxSteps = 1000;
  auto result = mode == 2
      ? propagator.propagate<decltype(options), DampedPerigeeReached>(start, target, options)
      : mode == 1
          ? propagator.propagate<decltype(options), Acts::ForcedSurfaceReached>(start, target, options)
          : propagator.propagate(start, target, options);
  if (!result.ok()) return {std::nullopt, 0., result.error().message()};
  if (!result->endParameters || (mode != 0 &&
      (!result->endParameters->parameters().allFinite() || !std::isfinite(result->pathLength)))) {
    return {std::nullopt, 0., "non-finite or missing endpoint parameters"};
  }
  return {std::move(result->endParameters), result->pathLength, ""};
}

inline TruthPerigeeResult truthPerigee(
    const Acts::GeometryContext& gctx, const Acts::MagneticFieldContext& mctx,
    const Acts::BoundTrackParameters& start,
    const std::shared_ptr<Acts::PerigeeSurface>& surface,
    std::shared_ptr<const Acts::MagneticFieldProvider> field,
    double maxR, double maxAbsZ, const Acts::Logger& logger) {
  auto legacy = truthPerigeeAttempt(gctx, mctx, start, *surface, field, logger, 0);
  // Recovery only: never validate away or recompute a legacy finite d0/z0 pair.
  if (legacy.parameters &&
      std::isfinite(legacy.parameters->parameters()[Acts::eBoundLoc0]) &&
      std::isfinite(legacy.parameters->parameters()[Acts::eBoundLoc1])) return legacy;

  auto recoveryField = std::make_shared<RecoveryField>(field, maxR, maxAbsZ);
  if (!recoveryField->contains(start.position(gctx))) {
    return {std::nullopt, 0., "production position outside recovery domain"};
  }
  TruthPerigeeResult result;
  for (unsigned int mode = 1; mode != 3; ++mode) {
    result = truthPerigeeAttempt(gctx, mctx, start, *surface, recoveryField, logger, mode);
    result.retries = mode;
    if (!result.parameters) continue;
    auto& at = *result.parameters;
    if (!recoveryField->contains(at.position(gctx))) {
      result = {std::nullopt, 0., "endpoint outside recovery domain"};
      continue;
    }
    const Acts::Vector3 relative = at.position(gctx) - surface->center(gctx);
    const Acts::Vector3 direction = at.direction();
    auto cache = recoveryField->makeCache(mctx);
    auto b = recoveryField->getField(at.position(gctx), cache);
    if (!b.ok()) { result = {std::nullopt, 0., b.error().message()}; continue; }
    const Acts::Vector3 acceleration = at.parameters()[Acts::eBoundQOverP] * direction.cross(*b);
    const double second = direction.head<2>().squaredNorm() + relative.head<2>().dot(acceleration.head<2>());
    if (!std::isfinite(second) || second <= 0) {
      result = {std::nullopt, 0., "target is not a transverse-distance minimum"};
      continue;
    }
    return result;
  }
  return result;
}

}  // namespace ActsExamples::detail
