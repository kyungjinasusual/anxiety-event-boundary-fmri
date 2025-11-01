# Code Monitor 📊

Comprehensive local code monitoring system for tracking contributions across Git repositories.

## Features

✨ **Smart Tracking**
- Weekly, monthly, or custom time periods
- Filter by author (yourself or team members)
- Track current branch or all branches
- Automatic name variation matching

📈 **Detailed Analytics**
- Commit counts and author statistics
- Code vs documentation file changes
- Line-by-line additions/deletions
- Per-repository breakdowns
- Branch statistics when using all-branches mode

🎨 **Multiple Output Formats**
- Colorful terminal output
- JSON export for automation
- Beautiful HTML reports with charts
- All formats can be combined

## Quick Start

### Basic Usage

```bash
# Your commits this week
./monitor.sh -a jub

# Weekly HTML report
./monitor.sh -a jub -h

# All branches (complete picture)
./monitor.sh -a jub -b -h

# Monthly review
./monitor.sh -w 4 -a jub -b -h
```

### Command Reference

```bash
./monitor.sh [OPTIONS]

Options:
  -a, --author NAME      Filter by author name (substring match, case-insensitive)
  -b, --all-branches     Include all branches (local + remote)
  -w, --weeks N          Number of weeks to analyze (default: 1)
  -h, --html             Generate HTML report
  -o, --output-dir DIR   Output directory for reports
  --help                 Show help message
```

## Features Explained

### 1. Author Filtering (`-a`)

Filter commits to show only specific author(s):

```bash
# Your commits only
./monitor.sh -a jub

# Matches any author containing "jub":
# - jub
# - jubilant-choi
# - Choi Jubin
```

**Finding your author name:**
```bash
git log --format='%an' | sort -u
```

### 2. All-Branches Mode (`-b`)

**Without `-b`** (default):
- Shows commits from current branch only
- Faster execution
- Good for daily checks

**With `-b`**:
- Shows commits from ALL branches (local + remote)
- Includes feature branches, development work
- Complete repository picture
- Finds work in unmerged branches

```bash
# Current branch only
./monitor.sh -a jub

# All branches (may find more commits!)
./monitor.sh -a jub -b
```

**Branch information displayed:**
```
🌿 Branches: 3 local, 36 remote (current: aurora)
```

### 3. Time Periods (`-w`)

```bash
-w 1    # Last week (default)
-w 2    # Last 2 weeks
-w 4    # Last month
-w 12   # Last quarter
```

### 4. Output Formats

**Terminal only** (fast):
```bash
./monitor.sh -a jub
```

**HTML report** (visual):
```bash
./monitor.sh -a jub -h
```

**JSON export** (automation):
```bash
python3 code_monitor.py -a jub --output data.json
```

**Combined** (everything):
```bash
./monitor.sh -a jub -h  # Creates both HTML and JSON
```

## Common Use Cases

### Daily Personal Check
```bash
./monitor.sh -a jub
```
Fast overview of your recent commits on current branch.

### Weekly Personal Review
```bash
./monitor.sh -a jub -b -h
```
Complete weekly report with HTML, all branches included.

### Monthly Portfolio Update
```bash
./monitor.sh -w 4 -a jub -b -h -o ~/portfolio/
```
Full month of your work, all branches, saved to portfolio directory.

### Team Weekly Meeting
```bash
./monitor.sh -b -h
```
Team activity across all branches with shareable HTML report.

### Find Missing Work
```bash
./monitor.sh -a jub -b
```
Discover your commits in feature branches that haven't been merged yet.

### Pre-Release Audit
```bash
./monitor.sh -b -w 8 -h
```
2-month audit across all branches before release.

## Automation

### Shell Aliases

Add to `~/.zshrc` or `~/.bashrc`:

```bash
# Personal tracking
alias mycode='./monitor.sh -a jub'
alias myweek='cd /Users/jub/Projects/code_monitor && ./monitor.sh -a jub -b -h'
alias mymonth='cd /Users/jub/Projects/code_monitor && ./monitor.sh -w 4 -a jub -b -h'

# Team tracking
alias teamweek='cd /Users/jub/Projects/code_monitor && ./monitor.sh -b -h'
```

Reload: `source ~/.zshrc`

Usage: `mycode`, `myweek`, `mymonth`, `teamweek`

### Cron Jobs

```bash
# Edit crontab
crontab -e

# Daily summary at 6 PM
0 18 * * * cd /Users/jub/Projects/code_monitor && ./monitor.sh -a jub > ~/daily_code.txt

# Weekly report every Monday at 9 AM
0 9 * * 1 cd /Users/jub/Projects/code_monitor && ./monitor.sh -a jub -b -h -o ~/Desktop/

# Monthly report on first Monday
0 9 1-7 * 1 cd /Users/jub/Projects/code_monitor && ./monitor.sh -w 4 -b -h -o ~/monthly/
```

## Python API

### Direct Script Usage

```bash
# Basic usage
python3 code_monitor.py --weeks 1

# With author filter
python3 code_monitor.py --author jub --weeks 1

# All branches
python3 code_monitor.py --author jub --all-branches --weeks 1

# Export to JSON
python3 code_monitor.py -a jub -b --weeks 4 --output monthly.json

# Generate HTML from JSON
python3 visualize_monitor.py monthly.json --output report.html
```

### Programmatic Usage

```python
from code_monitor import CodeMonitor

# Initialize
monitor = CodeMonitor(
    base_dir='/Users/jub/Projects',
    weeks=1,
    author_filter='jub',
    all_branches=True
)

# Analyze
results = monitor.analyze_repositories()

# Print report
monitor.print_report(results)

# Save JSON
monitor.save_json(results, 'output.json')

# Access data
for repo_name, data in results['repositories'].items():
    commits = data['file_changes']['commits']
    lines_added = data['line_changes']['lines_added']
    print(f"{repo_name}: {commits} commits, +{lines_added} lines")
```

## Configuration

Edit `monitor_config.yaml` to customize:

```yaml
# File categories
file_categories:
  code: [.py, .sh, .js, .ts, .java, .cpp, .go, ...]
  documentation: [.md, .txt, .rst, ...]
  configuration: [.yaml, .json, .toml, ...]

# Exclude patterns
exclude_patterns:
  - "**/node_modules/**"
  - "**/venv/**"
  - "**/__pycache__/**"
```

## Output Examples

### Terminal Output

```
======================================================================
📈 CODE MONITOR REPORT - 1 Week(s)
📅 Period: 2025-10-18 to 2025-10-25
👤 Author Filter: jub
🌿 Scope: All branches
======================================================================

📦 SwiFT_v2
   Commits: 2
   Authors: jubilant-choi
   🌿 Branches: 3 local, 36 remote (current: aurora)
   📝 Code Files:
      ✅ Added: 13
      ✏️  Modified: 11
   📚 Documentation:
      ✅ Added: 9
   📊 Lines: +6361 -117

📦 NeuroMamba
   Commits: 2
   Authors: jub, Choi Jubin
   🌿 Branches: 1 local, 2 remote (current: main)
   📝 Code Files:
      ✅ Added: 21
   📚 Documentation:
      ✅ Added: 1
   📊 Lines: +5068 -0

======================================================================
📊 SUMMARY
======================================================================
Total Commits: 10
Active Authors: 3

Code Files:
  ✅ Added: 45
  ✏️  Modified: 11

Documentation:
  ✅ Added: 19
  ✏️  Modified: 1

Total Lines: +16787 -136
Net Change: +16651 lines
======================================================================
```

### HTML Report

Beautiful gradient design with:
- Summary cards showing total statistics
- Repository cards with detailed breakdowns
- Visual progress bars for activity
- Responsive layout for all devices
- Author information and branch statistics

## Troubleshooting

### No data showing?

**Check 1**: Verify time range
```bash
# Try longer period
./monitor.sh -a jub -w 2
```

**Check 2**: Verify author name
```bash
# See all author names in repos
for repo in */; do
    [ -d "$repo/.git" ] && (cd "$repo" && git log --format='%an' | sort -u)
done | sort -u
```

**Check 3**: Try all branches
```bash
# Your work might be in other branches
./monitor.sh -a jub -b
```

**Check 4**: Ensure Git repositories
```bash
ls -la | grep .git
```

### Different numbers than expected?

- **Without `-b`**: Shows only current branch
- **With `-b`**: Shows ALL branches (more commits likely)
- Commits in feature branches won't show without `-b`

### Scripts won't run?

```bash
chmod +x code_monitor.py visualize_monitor.py monitor.sh
```

### Want to track different directory?

```bash
python3 code_monitor.py --dir /path/to/repos --weeks 1
```

## Performance

**Fast operations** (~2-5 seconds per repo):
- Current branch only
- Short time periods (1-2 weeks)
- Single author filter

**Slower but still fast** (~5-10 seconds per repo):
- All branches mode (depends on branch count)
- Long time periods (12+ weeks)
- Many repositories

## Files

- `code_monitor.py` - Main analysis script
- `monitor.sh` - Convenient wrapper script
- `visualize_monitor.py` - HTML report generator
- `monitor_config.yaml` - Configuration file
- `README.md` - This file
- `QUICKSTART.md` - Quick reference

## Requirements

- Python 3.6+
- Git
- No external Python packages (uses only stdlib)

## Tips & Best Practices

1. **Daily**: Use `-a jub` for quick personal checks
2. **Weekly**: Use `-a jub -b -h` for complete personal review
3. **Monthly**: Use `-w 4 -b -h` for team/management reports
4. **Lost work**: Use `-a jub -b` to find work in unmerged branches
5. **Automation**: Set up aliases and cron jobs for regular reports

## Command Cheat Sheet

| Scenario | Command |
|----------|---------|
| Quick personal check | `./monitor.sh -a jub` |
| Weekly personal HTML | `./monitor.sh -a jub -h` |
| Complete personal review | `./monitor.sh -a jub -b -h` |
| Monthly review | `./monitor.sh -w 4 -a jub -b -h` |
| Team weekly | `./monitor.sh -b -h` |
| Find lost work | `./monitor.sh -a jub -b` |
| Pre-release audit | `./monitor.sh -b -w 8 -h` |
| Export to JSON | `python3 code_monitor.py -a jub --output data.json` |

## License

Free to use and modify for personal and commercial projects.

---

**Pro Tip**: For complete visibility, use `./monitor.sh -a jub -b -h` 🚀
