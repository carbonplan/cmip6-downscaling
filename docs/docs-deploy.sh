#!/bin/bash

if [[ "$VERCEL_ENV" == production || "$VERCEL_GIT_COMMIT_MESSAGE" == *"[docs]"* ]] ; then
  # Proceed with the build
	echo "✅ - Build can proceed"
  exit 1;

else
  # Don't build
  echo "🛑 - Build cancelled"
  exit 0;
fi
