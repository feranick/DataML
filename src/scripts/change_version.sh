#!/bin/bash
####################################
# change_version.sh
# Change version tag in any file
#  contained in the specified folder
# By Nicola Ferralis - 2026.03.06.3
# ##################################

# Ensure both the old and new version arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <old_version> <new_version> [target_directory]"
    exit 1
fi

OLD_VERSION="$1"
NEW_VERSION="$2"
TARGET_DIR="${3:-.}" # Defaults to the current directory if a 3rd argument isn't provided

# Escape periods in the old version string so sed treats them literally
ESCAPED_OLD=$(printf '%s\n' "$OLD_VERSION" | sed 's/\./\\./g')

# Detect the operating system to set the correct in-place flag for sed
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS / BSD sed requires an empty string for the backup extension
    SED_INPLACE=(-i '')
else
    # Linux / GNU sed does not want the empty string
    SED_INPLACE=(-i)
fi

# Find all regular files (excluding hidden directories) and perform the replacement
find "$TARGET_DIR" -type f -not -path '*/\.*' -exec sed "${SED_INPLACE[@]}" "s/${ESCAPED_OLD}/${NEW_VERSION}/g" {} +

echo "Successfully replaced '$OLD_VERSION' with '$NEW_VERSION' in '$TARGET_DIR'."
