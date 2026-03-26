# CODEOWNERS for Orbit-Q

# Default owner (fallback)
*                                   @poojakira

# ML / MLOps ownership → Pooja Kiran
/src/orbit_q/engine/                @poojakira
/src/orbit_q/engine/models/         @poojakira
/src/orbit_q/engine/kernels/        @poojakira
/src/orbit_q/mlflow_tracking/       @poojakira
/tests/test_ml_engine.py            @poojakira

# Robotics / mission-control ownership → Rhutvik Pachghare
/src/orbit_q/simulator/             @Rhutvik-Pachghare
/src/orbit_q/orchestrator/          @Rhutvik-Pachghare
/src/orbit_q/dashboard/             @Rhutvik-Pachghare
/tests/test_simulator.py            @Rhutvik-Pachghare
/tests/test_security_and_stress.py  @Rhutvik-Pachghare
