#!/bin/bash

echo "VERCEL_ENV: $VERCEL_ENV"
echo "VERCEL_GIT_COMMIT_MESSAGE: $VERCEL_GIT_COMMIT_MESSAGE"


if [[ "$VERCEL_ENV" == production || "$VERCEL_GIT_COMMIT_MESSAGE" == *[docs]* ]] ; then
  # Proceed with the build
	echo "✅ - Build would have proceeded"
  exit 0;

else
  # Don't build
  echo "🛑 - Build would have been cancelled"
  exit 0;
fi
