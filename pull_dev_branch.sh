#!/bin/bash

# Exit on error
set -e

# Make sure we're on the dev branch
git checkout dev

# Pull the latest changes
git pull origin dev

echo "Repository successfully updated from dev branch."