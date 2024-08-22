# MLFlow configuration
# Create an MLFlow run and put it into our final configuration.
# When we launch a training on GCP, each node that we have in our instance group can read this run ID and log their parameters and metric values to the same MLFlow run.
# Later we will use this final configuration to build a Docker image and push it to GCP Artifact Repository.
#   The VM instances will access this configurations and use the run_id to log their parameters and metric values.

from pathlib import Path
from typing import TYPE_CHECKING

import mlflow

from omegaconf import OmegaConf
from cybulde.utils.config_utils import get_config, save_config_as_yaml
from cybulde.utils.mlflow_utils import activate_mlflow

# TYPE_CHECKING is used to conditionally include code that is only relevant for type checking tools (like mypy) and not for runtime execution.
# At runtime, TYPE_CHECKING is always False, so the code inside the block will not be executed.
if TYPE_CHECKING:
  from cybulde.config_schemas.config_schema import Config


@get_config(config_path="../configs", config_name="config")
def generate_final_config(config: "Config"):
  # print(OmegaConf.to_yaml(config))
  # Helper function to activate the MLFlow run.
  with activate_mlflow(
        config.infrastructure.mlflow.experiment_name,
        run_id=config.infrastructure.mlflow.run_id,
        run_name=config.infrastructure.mlflow.run_name,
    ) as run:

    # Get the run_id, experiment_id and artifact_uri from the run.
    run_id: str = run.info.run_id
    experiment_id: str = run.info.experiment_id
    artifact_uri: str = run.info.artifact_uri

    # Put run_id, experiment_id and artifact_uri into our final config.
    config.infrastructure.mlflow.artifact_uri = artifact_uri
    config.infrastructure.mlflow.run_id = run_id
    config.infrastructure.mlflow.experiment_id = experiment_id

    # Save our configuration
    config_save_dir = Path("./cybulde/configs/automatically_generated/")
    # Create /automatically_generated folder
    config_save_dir.mkdir(parents=True, exist_ok=True)
    # Create __init__ files.
    (config_save_dir / "__init__.py").touch(exist_ok=True)
  
    yaml_config_save_path = config_save_dir / "config.yaml"
    save_config_as_yaml(config, str(yaml_config_save_path))

    # Log it to MLFlow
    mlflow.log_artifact(str(yaml_config_save_path))

    # log_training_hparams(config)
    # log_artifacts_for_reproducibility()


# Definition of where our MLFlow configuration will reside.

if __name__ == "__main__":
  generate_final_config()