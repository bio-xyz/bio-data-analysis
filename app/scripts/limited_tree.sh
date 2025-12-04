#!/bin/bash

# Usage: ./limited_tree.sh [directory] [max_items_per_dir] [max_depth] [max_total_lines]
# Defaults: current dir, 20 items, depth 10, 200 total lines

DIR="${1:-.}"
MAX_ITEMS="${2:-20}"
MAX_DEPTH="${3:-10}"
MAX_LINES="${4:-200}"

LINE_COUNT=0
STOPPED_EARLY=0

print_tree() {
    local dir="$1"
    local prefix="$2"
    local depth="$3"
    
    # Stop if we've hit max lines
    if [ "$LINE_COUNT" -ge "$MAX_LINES" ]; then
        if [ "$STOPPED_EARLY" -eq 0 ]; then
            echo "... (output truncated at $MAX_LINES lines)"
            STOPPED_EARLY=1
        fi
        return
    fi
    
    # Stop if we've reached max depth
    if [ "$depth" -ge "$MAX_DEPTH" ]; then
        return
    fi
    
    # Get all entries (directories first, then files)
    local dirs=()
    local files=()
    
    while IFS= read -r -d '' entry; do
        if [ -d "$entry" ]; then
            dirs+=("$entry")
        else
            files+=("$entry")
        fi
    done < <(find "$dir" -maxdepth 1 -mindepth 1 -print0 2>/dev/null | sort -z)
    
    local total=$((${#dirs[@]} + ${#files[@]}))
    local shown=0
    
    # Show directories first
    for entry in "${dirs[@]}"; do
        if [ "$LINE_COUNT" -ge "$MAX_LINES" ]; then
            if [ "$STOPPED_EARLY" -eq 0 ]; then
                echo "... (output truncated at $MAX_LINES lines)"
                STOPPED_EARLY=1
            fi
            return
        fi
        
        if [ "$shown" -ge "$MAX_ITEMS" ]; then
            local remaining=$((total - shown))
            echo "${prefix}... and $remaining more items"
            ((LINE_COUNT++))
            return
        fi
        
        local name=$(basename "$entry")
        echo "${prefix}${name}/"
        ((LINE_COUNT++))
        ((shown++))
        
        print_tree "$entry" "${prefix}    " $((depth + 1))
    done
    
    # Show files
    for entry in "${files[@]}"; do
        if [ "$LINE_COUNT" -ge "$MAX_LINES" ]; then
            if [ "$STOPPED_EARLY" -eq 0 ]; then
                echo "... (output truncated at $MAX_LINES lines)"
                STOPPED_EARLY=1
            fi
            return
        fi
        
        if [ "$shown" -ge "$MAX_ITEMS" ]; then
            local remaining=$((total - shown))
            echo "${prefix}... and $remaining more items"
            ((LINE_COUNT++))
            return
        fi
        
        local name=$(basename "$entry")
        echo "${prefix}${name}"
        ((LINE_COUNT++))
        ((shown++))
    done
}

echo "$DIR"
((LINE_COUNT++))
print_tree "$DIR" "    " 0
