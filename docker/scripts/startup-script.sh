#!/usr/bin/env bash

# As this startup script is going to be used both by our production and local development service, we need an IF statement in order to check which one we are running and do something based on that.

set -o errexit
set -o pipefail
set -o nounset

if [[ "${IS_PROD_ENV}" == "true" ]]; then
	/usr/local/gcloud/google-cloud-sdk/bin/gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --tunnel-through-iap -- -4 -N -L ${PROD_MLFLOW_SERVER_PORT}:localhost:${PROD_MLFLOW_SERVER_PORT}
else
	# /start-prediction-service.sh &
	/start-tracking-server.sh & # Run our local MLFlow service.
	tail -F anything
fi
