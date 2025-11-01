# Code Monitor - Quick Start 🚀

Track your code contributions across all GitHub repositories locally.

## Installation

No installation needed! All scripts are ready to use in `/Users/jub/Projects/code_monitor/`

## Most Common Commands

### Daily Quick Check
```bash
./monitor.sh -a jub
```

### Weekly Report with HTML
```bash
./monitor.sh -a jub -h
```

### Monthly Review
```bash
./monitor.sh -w 4 -a jub -h
```

## Quick Reference

| What you want | Command |
|---------------|---------|
| Your commits this week | `./monitor.sh -a jub` |
| Weekly HTML report | `./monitor.sh -a jub -h` |
| Complete monthly report | `./monitor.sh -w 4 -a jub -h` |
| Team activity | `./monitor.sh -h` |

## Flags

- `-a NAME` or `--author NAME`: Filter by author (e.g., `-a jub`)
- `-w N` or `--weeks N`: Number of weeks to analyze (default: 1)
- `-h` or `--html`: Generate HTML report
- `-o DIR` or `--output-dir DIR`: Output directory

## Setup Aliases (Optional)

Add to `~/.zshrc`:

```bash
alias mycode='./monitor.sh -a jub'
alias myweek='cd /Users/jub/Projects/code_monitor && ./monitor.sh -a jub -h'
alias mymonth='cd /Users/jub/Projects/code_monitor && ./monitor.sh -w 4 -a jub -h'
```

Then reload: `source ~/.zshrc`

Now just type: `mycode`, `myweek`, or `mymonth`

## What You'll See

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
   📝 Code Files: ✅ 13 added, ✏️ 11 modified
   📚 Documentation: ✅ 9 added
   📊 Lines: +6361 -117

======================================================================
📊 SUMMARY
======================================================================
Total Commits: 10
Active Authors: 3
Code Files: ✅ 45 added, ✏️ 11 modified
Documentation: ✅ 19 added, ✏️ 1 modified
Total Lines: +16787 -136
Net Change: +16651 lines
======================================================================
```

## Common Scenarios

**Daily standup prep:**
```bash
./monitor.sh -a jub
```

**Weekly personal review:**
```bash
./monitor.sh -a jub -h
```

**Monthly portfolio update:**
```bash
./monitor.sh -w 4 -a jub -h -o ~/portfolio/
```

**Team weekly meeting:**
```bash
./monitor.sh -h
```

## Help

```bash
./monitor.sh --help
python3 code_monitor.py --help
```

See `README.md` for complete documentation.

---

**Most used**: `./monitor.sh -a jub -h` 📊✨
