#!/bin/bash
####################################
# change_version.sh
# Change version tag in any file
#  contained in the specified folder
# By Nicola Ferralis - 2026.03.06.3
####################################

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

# Set LC_ALL to C to prevent illegal byte sequence errors on macOS
export LC_ALL=C

# Initialize counter
CHANGED_COUNT=0

echo "Scanning '$TARGET_DIR' for files containing '$OLD_VERSION'..."
echo "--------------------------------------------------------------"

# Use process substitution to feed the loop, keeping CHANGED_COUNT in the current shell
while IFS= read -r file; do
    # Perform the replacement
    sed "${SED_INPLACE[@]}" "s/${ESCAPED_OLD}/${NEW_VERSION}/g" "$file"
    
    # Print the updated file and increment the counter
    echo "Updated: $file"
    ((CHANGED_COUNT++))
done < <(find "$TARGET_DIR" -type f -not -path '*/\.*' -exec grep -Il "$OLD_VERSION" {} +)

echo "--------------------------------------------------------------"
echo "Success: Replaced '$OLD_VERSION' with '$NEW_VERSION' in $CHANGED_COUNT file(s)."
