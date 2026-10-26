ARG BASE_IMAGE=colliderml-acts:7cba36b17
FROM ${BASE_IMAGE}
USER root
ARG JOBS=1
COPY writer.patch CMakeLists.txt PerigeePatchVersion.hpp.in /opt/colliderml-perigee/
COPY include/ /opt/acts-arrow-src/Examples/Io/Root/include/
RUN test "$(git -C /opt/acts-arrow-src rev-parse HEAD)" = 7cba36b173dd73a14342cc65043a43179a4b1dbd \
 && git -C /opt/acts-arrow-src restore --source=HEAD -- Examples/Io/Root/src/RootParticleWriter.cpp Examples/Io/Root/include/ActsExamples/Io/Root/RootParticleWriter.hpp Python/Examples/src/plugins/Root.cpp \
 && git -C /opt/acts-arrow-src apply --check /opt/colliderml-perigee/writer.patch \
 && git -C /opt/acts-arrow-src apply /opt/colliderml-perigee/writer.patch
RUN /bin/bash -c 'set -euo pipefail; \
    source /opt/acts-arrow/setup.sh; \
    cmake -S /opt/colliderml-perigee -B /tmp/acts-perigee-build \
      -DCMAKE_INSTALL_PREFIX=/opt/acts-arrow \
      -DPython_EXECUTABLE="$(command -v python3)" \
      -DCMAKE_PREFIX_PATH="$(cat /opt/acts-arrow-cmake-prefix.txt)"; \
    cmake --build /tmp/acts-perigee-build --parallel '"${JOBS}"'; \
    cmake --install /tmp/acts-perigee-build; \
    rm -rf /tmp/acts-perigee-build'
RUN /bin/bash -c 'set -e; source /opt/acts-arrow/setup.sh; \
    python3 -c "import acts.examples.root as r; c=r.RootParticleWriter.Config(); c.perigeeRecoveryMaxR=1160.; c.perigeeRecoveryMaxAbsZ=3000.; assert not hasattr(c,\"perigeeFieldBoundaryZ\")"'
LABEL colliderml.perigee.patch="writer-recovery-v2" \
      colliderml.acts.base="7cba36b173dd73a14342cc65043a43179a4b1dbd"
