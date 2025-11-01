#!/bin/bash
# Code Monitor - Convenient wrapper script

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
WEEKS=1
OUTPUT_DIR="."
GENERATE_HTML=false
AUTHOR=""
ALL_BRANCHES=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -w|--weeks)
            WEEKS="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -a|--author)
            AUTHOR="$2"
            shift 2
            ;;
        -h|--html)
            GENERATE_HTML=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -w, --weeks N        Number of weeks to analyze (default: 1)"
            echo "  -a, --author NAME    Filter by author name (case-insensitive)"
            echo "  -o, --output-dir DIR Output directory for reports (default: current dir)"
            echo "  -h, --html           Generate HTML report"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                          # Analyze last week, terminal output only"
            echo "  $0 -w 4                     # Analyze last 4 weeks (monthly)"
            echo "  $0 -a jub                   # Only commits by 'jub'"
            echo "  $0 -b                       # All branches in all repos"
            echo "  $0 -a jub -b                # Your commits across all branches"
            echo "  $0 -w 1 -a jub -b -h        # Full report: you, all branches, HTML"
            echo "  $0 -w 2 -o ~/reports -h     # Custom output directory"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create timestamp for filenames
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JSON_FILE="${OUTPUT_DIR}/code_monitor_${TIMESTAMP}.json"

echo -e "${BLUE}📊 Code Monitor${NC}"
echo -e "${BLUE}=================${NC}"
echo ""

# Build command arguments
PYTHON_ARGS="--weeks $WEEKS"
if [ -n "$AUTHOR" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --author \"$AUTHOR\""
fi
if [ "$ALL_BRANCHES" = true ]; then
    PYTHON_ARGS="$PYTHON_ARGS --all-branches"
fi

# Build description for output
DESCRIPTION="last ${WEEKS} week(s)"
if [ -n "$AUTHOR" ]; then
    DESCRIPTION="$DESCRIPTION by author '${AUTHOR}'"
fi
if [ "$ALL_BRANCHES" = true ]; then
    DESCRIPTION="$DESCRIPTION (all branches)"
fi

# Run the monitor
if [ "$GENERATE_HTML" = true ]; then
    echo -e "${GREEN}→ Generating report for ${DESCRIPTION} with HTML output...${NC}"
    eval python3 "${SCRIPT_DIR}/code_monitor.py" $PYTHON_ARGS --output "$JSON_FILE"

    if [ -f "$JSON_FILE" ]; then
        HTML_FILE="${OUTPUT_DIR}/code_monitor_${TIMESTAMP}.html"
        python3 "${SCRIPT_DIR}/visualize_monitor.py" "$JSON_FILE" --output "$HTML_FILE"

        echo ""
        echo -e "${GREEN}✅ Reports generated:${NC}"
        echo -e "   JSON: ${JSON_FILE}"
        echo -e "   HTML: ${HTML_FILE}"
        echo ""
        echo -e "${GREEN}→ Opening HTML report...${NC}"

        # Try to open in browser
        if command -v open &> /dev/null; then
            open "$HTML_FILE"
        elif command -v xdg-open &> /dev/null; then
            xdg-open "$HTML_FILE"
        else
            echo "   Open this file in your browser: ${HTML_FILE}"
        fi
    fi
else
    echo -e "${GREEN}→ Analyzing ${DESCRIPTION}...${NC}"
    python3 "${SCRIPT_DIR}/code_monitor.py" $PYTHON_ARGS
fi
