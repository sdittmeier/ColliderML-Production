// Diagnostic only: reproduce the pinned RootParticleWriter propagation.
#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/MagneticField/MagneticFieldContext.hpp"
#include "Acts/MagneticField/MagneticFieldProvider.hpp"
#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Propagator/SympyStepper.hpp"
#include "Acts/Surfaces/PerigeeSurface.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "ActsExamples/Io/Root/detail/TruthPerigee.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

struct DampedPerigeeReached : Acts::ForcedSurfaceReached {
  template <typename State, typename Stepper, typename Navigator>
  bool checkAbort(State& state, const Stepper& stepper,
                  const Navigator& navigator, const Acts::Logger& logger) const {
    const bool reached = Acts::ForcedSurfaceReached::checkAbort(state, stepper, navigator, logger);
    if (!reached && surface != nullptr) {
      auto intersection = surface->intersect(
          state.geoContext, stepper.position(state.stepping),
          state.options.direction * stepper.direction(state.stepping),
          boundaryTolerance, state.options.surfaceTolerance).closest();
      if (intersection.isValid()) {
        stepper.updateStepSize(state.stepping, 0.5 * intersection.pathLength(),
                               Acts::ConstrainedStep::Type::Actor);
      }
    }
    return reached;
  }
};

template <typename Stepper>
py::dict propagate(std::shared_ptr<const Acts::MagneticFieldProvider> field,
                   const Acts::Vector3& position,
                   const Acts::Vector3& momentum, double charge,
                   int directionOverride, bool loopProtection,
                   unsigned int maxSteps, bool forcedTarget,
                   double surfaceTolerance, double maxStepSize, bool verbose,
                   bool dampedTarget) {
  Acts::GeometryContext gctx;
  Acts::MagneticFieldContext mctx;
  const auto direction = momentum.normalized().eval();
  auto surface = Acts::Surface::makeShared<Acts::PerigeeSurface>(Acts::Vector3::Zero());
  auto intersection = surface->intersect(gctx, position, direction,
                                        Acts::BoundaryTolerance::Infinite(), surfaceTolerance).closest();
  auto start = Acts::BoundTrackParameters::createCurvilinear(
      Acts::Vector4(position.x(), position.y(), position.z(), 0.), direction,
      charge / momentum.norm(), std::nullopt, Acts::ParticleHypothesis::pion());
  using Propagator = Acts::Propagator<Stepper>;
  Propagator propagator(Stepper(field), typename Propagator::Navigator{},
                        Acts::getDefaultLogger("PerigeeProbe", verbose ? Acts::Logging::VERBOSE : Acts::Logging::WARNING));
  typename Propagator::template Options<> options(gctx, mctx);
  const auto selected = Acts::Direction::fromScalarZeroAsPositive(intersection.pathLength());
  options.direction = directionOverride == 0 ? selected : Acts::Direction::fromScalarZeroAsPositive(directionOverride);
  options.loopProtection = loopProtection;
  options.maxSteps = maxSteps;
  options.surfaceTolerance = surfaceTolerance;
  if (maxStepSize > 0) {
    options.stepping.maxStepSize = maxStepSize;
  }
  auto result = dampedTarget
      ? propagator.template propagate<decltype(options), DampedPerigeeReached>(start, *surface, options)
      : forcedTarget
          ? propagator.template propagate<decltype(options), Acts::ForcedSurfaceReached>(start, *surface, options)
          : propagator.propagate(start, *surface, options);
  py::dict out;
  out["linear_intersection_path_mm"] = intersection.pathLength();
  out["ok"] = result.ok();
  if (!result.ok()) {
    out["error"] = result.error().message();
    out["error_code"] = result.error().value();
    out["error_category"] = result.error().category().name();
  } else {
    out["has_end_parameters"] = result->endParameters.has_value();
    out["path_length_mm"] = result->pathLength;
    if (result->endParameters) {
      out["d0"] = result->endParameters->parameters()[Acts::eBoundLoc0];
      out["z0"] = result->endParameters->parameters()[Acts::eBoundLoc1];
      out["finite_parameters"] = result->endParameters->parameters().allFinite();
    }
  }
  return out;
}

PYBIND11_MODULE(perigee_probe, module) {
  module.def("writer_perigee", [](std::shared_ptr<const Acts::MagneticFieldProvider> field,
                                   const Acts::Vector3& position, const Acts::Vector3& momentum,
                                   double charge, double maxR, double maxAbsZ) {
    auto gctx = Acts::GeometryContext::dangerouslyDefaultConstruct();
    Acts::MagneticFieldContext mctx;
    auto start = Acts::BoundTrackParameters::createCurvilinear(
        Acts::Vector4(position.x(), position.y(), position.z(), 0.), momentum.normalized(),
        charge / momentum.norm(), std::nullopt, Acts::ParticleHypothesis::pion());
    auto surface = Acts::Surface::makeShared<Acts::PerigeeSurface>(Acts::Vector3::Zero());
    auto logger = Acts::getDefaultLogger("WriterPerigeeTest", Acts::Logging::WARNING);
    auto result = ActsExamples::detail::truthPerigee(gctx, mctx, start, surface, field, maxR, maxAbsZ, *logger);
    py::dict out;
    out["ok"] = result.parameters.has_value();
    out["error"] = result.error;
    out["retries"] = result.retries;
    out["boundaries"] = result.boundaries;
    if (result.parameters) {
      out["has_end_parameters"] = true;
      out["finite_parameters"] = result.parameters->parameters().allFinite();
      out["d0"] = result.parameters->parameters()[Acts::eBoundLoc0];
      out["z0"] = result.parameters->parameters()[Acts::eBoundLoc1];
      out["path_length_mm"] = result.pathLength;
    }
    return out;
  }, py::arg("field"), py::arg("position"), py::arg("momentum"), py::arg("charge"),
     py::arg("recovery_max_r") = 1160., py::arg("recovery_max_abs_z") = 3000.);
  module.def("field_tesla", [](std::shared_ptr<const Acts::MagneticFieldProvider> field,
                               const Acts::Vector3& position) {
    auto cache = field->makeCache(Acts::MagneticFieldContext{});
    auto result = field->getField(position, cache);
    if (!result.ok()) {
      throw std::runtime_error(result.error().message());
    }
    return (result.value() / Acts::UnitConstants::T).eval();
  });
  module.def("sympy", &propagate<Acts::SympyStepper>, py::arg("field"),
             py::arg("position"), py::arg("momentum"), py::arg("charge"),
             py::arg("direction") = 0, py::arg("loop_protection") = true,
             py::arg("max_steps") = 1000, py::arg("forced_target") = false,
             py::arg("surface_tolerance") = 1e-4, py::arg("max_step_size") = -1., py::arg("verbose") = false,
             py::arg("damped_target") = false);
  module.def("eigen", &propagate<Acts::EigenStepper<>>, py::arg("field"),
             py::arg("position"), py::arg("momentum"), py::arg("charge"),
             py::arg("direction") = 0, py::arg("loop_protection") = true,
             py::arg("max_steps") = 1000, py::arg("forced_target") = false,
             py::arg("surface_tolerance") = 1e-4, py::arg("max_step_size") = -1., py::arg("verbose") = false,
             py::arg("damped_target") = false);
}
