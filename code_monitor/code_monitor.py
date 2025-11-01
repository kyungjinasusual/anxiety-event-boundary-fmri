#!/usr/bin/env python3
"""
Weekly Code Monitor - Track code and documentation changes across Git repositories
"""

import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import json
import argparse


class CodeMonitor:
    """Monitor code generation and changes across Git repositories"""

    # File categories
    CODE_EXTENSIONS = {'.py', '.sh', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.rb', '.php'}
    DOC_EXTENSIONS = {'.md', '.txt', '.rst', '.adoc', '.org'}
    CONFIG_EXTENSIONS = {'.yaml', '.yml', '.json', '.toml', '.ini', '.conf', '.xml'}

    def __init__(self, base_dir: str, weeks: int = 1, author_filter: str = None, all_branches: bool = False):
        self.base_dir = Path(base_dir).resolve()
        self.weeks = weeks
        self.author_filter = author_filter
        self.all_branches = all_branches
        self.repos = self._find_repositories()

    def _find_repositories(self) -> list:
        """Find all Git repositories in the base directory"""
        repos = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and (item / '.git').exists():
                repos.append(item)
        return sorted(repos)

    def _run_git_command(self, repo_path: Path, command: list) -> str:
        """Execute Git command in repository"""
        try:
            result = subprocess.run(
                ['git'] + command,
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Git error in {repo_path.name}: {e.stderr}")
            return ""

    def _get_file_stats(self, repo_path: Path, since_date: str) -> dict:
        """Get file statistics for a repository since date"""
        stats = {
            'code': defaultdict(lambda: {'added': 0, 'modified': 0, 'deleted': 0}),
            'docs': defaultdict(lambda: {'added': 0, 'modified': 0, 'deleted': 0}),
            'config': defaultdict(lambda: {'added': 0, 'modified': 0, 'deleted': 0}),
            'other': defaultdict(lambda: {'added': 0, 'modified': 0, 'deleted': 0}),
            'commits': 0,
            'authors': set()
        }

        # Build Git log command with optional author filter and all-branches
        log_command = ['log', f'--since={since_date}', '--pretty=format:%H|%an', '--name-status']
        if self.all_branches:
            log_command.append('--all')
        if self.author_filter:
            log_command.append(f'--author={self.author_filter}')

        # Get commit log
        log_output = self._run_git_command(repo_path, log_command)

        if not log_output:
            return stats

        current_commit = None
        current_author = None
        skip_files = False

        for line in log_output.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Parse commit line
            if '|' in line:
                commit_hash, author = line.split('|', 1)
                current_author = author

                # Check if we should process this commit
                skip_files = False
                if self.author_filter and self.author_filter.lower() not in author.lower():
                    skip_files = True
                    continue

                stats['authors'].add(author)
                stats['commits'] += 1
                current_commit = commit_hash
                continue

            # Skip files if author doesn't match
            if skip_files:
                continue

            # Parse file change line (A/M/D filename)
            parts = line.split('\t')
            if len(parts) < 2:
                continue

            status = parts[0]
            filepath = parts[1]
            ext = Path(filepath).suffix.lower()

            # Categorize file
            if ext in self.CODE_EXTENSIONS:
                category = 'code'
            elif ext in self.DOC_EXTENSIONS:
                category = 'docs'
            elif ext in self.CONFIG_EXTENSIONS:
                category = 'config'
            else:
                category = 'other'

            # Track change type
            if status == 'A':
                stats[category][ext]['added'] += 1
            elif status == 'M':
                stats[category][ext]['modified'] += 1
            elif status == 'D':
                stats[category][ext]['deleted'] += 1

        # Convert authors set to count
        stats['author_count'] = len(stats['authors'])
        stats['authors'] = list(stats['authors'])

        return stats

    def _get_line_stats(self, repo_path: Path, since_date: str) -> dict:
        """Get line addition/deletion statistics"""
        # Build Git log command with optional author filter and all-branches
        log_command = ['log', f'--since={since_date}', '--numstat', '--pretty=format:']
        if self.all_branches:
            log_command.append('--all')
        if self.author_filter:
            log_command.append(f'--author={self.author_filter}')

        output = self._run_git_command(repo_path, log_command)

        stats = {
            'lines_added': 0,
            'lines_deleted': 0,
            'code_lines_added': 0,
            'code_lines_deleted': 0,
            'doc_lines_added': 0,
            'doc_lines_deleted': 0
        }

        for line in output.split('\n'):
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) >= 3:
                try:
                    added = int(parts[0]) if parts[0] != '-' else 0
                    deleted = int(parts[1]) if parts[1] != '-' else 0
                    filepath = parts[2]
                    ext = Path(filepath).suffix.lower()

                    stats['lines_added'] += added
                    stats['lines_deleted'] += deleted

                    # Categorize by file type
                    if ext in self.CODE_EXTENSIONS:
                        stats['code_lines_added'] += added
                        stats['code_lines_deleted'] += deleted
                    elif ext in self.DOC_EXTENSIONS:
                        stats['doc_lines_added'] += added
                        stats['doc_lines_deleted'] += deleted
                except ValueError:
                    continue

        return stats

    def _get_branch_info(self, repo_path: Path) -> dict:
        """Get branch information for repository"""
        # Get all branches
        all_branches_output = self._run_git_command(repo_path, ['branch', '-a'])
        local_branches = [b.strip().replace('* ', '') for b in all_branches_output.split('\n')
                         if b.strip() and not b.strip().startswith('remotes/')]

        # Get remote branches
        remote_branches_output = self._run_git_command(repo_path, ['branch', '-r'])
        remote_branches = [b.strip() for b in remote_branches_output.split('\n') if b.strip()]

        # Get current branch
        current_branch = self._run_git_command(repo_path, ['rev-parse', '--abbrev-ref', 'HEAD']).strip()

        return {
            'current': current_branch,
            'local_count': len(local_branches),
            'remote_count': len(remote_branches),
            'local_branches': local_branches[:10],  # Limit for display
            'remote_branches': remote_branches[:10]
        }

    def analyze_repositories(self) -> dict:
        """Analyze all repositories for the specified time period"""
        since_date = (datetime.now() - timedelta(weeks=self.weeks)).strftime('%Y-%m-%d')

        results = {
            'period': {
                'weeks': self.weeks,
                'since': since_date,
                'until': datetime.now().strftime('%Y-%m-%d'),
                'author_filter': self.author_filter,
                'all_branches': self.all_branches
            },
            'repositories': {}
        }

        for repo in self.repos:
            print(f"📊 Analyzing {repo.name}...")

            file_stats = self._get_file_stats(repo, since_date)
            line_stats = self._get_line_stats(repo, since_date)
            branch_info = self._get_branch_info(repo) if self.all_branches else None

            results['repositories'][repo.name] = {
                'file_changes': file_stats,
                'line_changes': line_stats,
                'branch_info': branch_info
            }

        return results

    def _calculate_totals(self, category_data: dict) -> dict:
        """Calculate totals for a category"""
        return {
            'added': sum(v['added'] for v in category_data.values()),
            'modified': sum(v['modified'] for v in category_data.values()),
            'deleted': sum(v['deleted'] for v in category_data.values())
        }

    def print_report(self, results: dict):
        """Print formatted report"""
        period = results['period']

        print("\n" + "="*70)
        print(f"📈 CODE MONITOR REPORT - {period['weeks']} Week(s)")
        print(f"📅 Period: {period['since']} to {period['until']}")
        if period.get('author_filter'):
            print(f"👤 Author Filter: {period['author_filter']}")
        if period.get('all_branches'):
            print(f"🌿 Scope: All branches")
        else:
            print(f"🌿 Scope: Current branch only")
        print("="*70 + "\n")

        total_commits = 0
        total_code_added = 0
        total_code_modified = 0
        total_docs_added = 0
        total_docs_modified = 0
        total_lines_added = 0
        total_lines_deleted = 0
        total_code_lines_added = 0
        total_code_lines_deleted = 0
        total_doc_lines_added = 0
        total_doc_lines_deleted = 0
        all_authors = set()

        for repo_name, data in results['repositories'].items():
            file_changes = data['file_changes']
            line_changes = data['line_changes']

            if file_changes['commits'] == 0:
                continue

            print(f"📦 {repo_name}")
            print(f"   Commits: {file_changes['commits']}")
            print(f"   Authors: {', '.join(file_changes['authors'][:3])}" +
                  (f" +{len(file_changes['authors']) - 3} more" if len(file_changes['authors']) > 3 else ""))

            # Branch information (if all_branches mode)
            if data.get('branch_info'):
                branch_info = data['branch_info']
                print(f"   🌿 Branches: {branch_info['local_count']} local, {branch_info['remote_count']} remote (current: {branch_info['current']})")

            # Code files
            code_totals = self._calculate_totals(file_changes['code'])
            if any(code_totals.values()):
                print(f"   📝 Code Files:")
                print(f"      ✅ Added: {code_totals['added']}")
                print(f"      ✏️  Modified: {code_totals['modified']}")
                print(f"      ❌ Deleted: {code_totals['deleted']}")
                total_code_added += code_totals['added']
                total_code_modified += code_totals['modified']

            # Documentation files
            docs_totals = self._calculate_totals(file_changes['docs'])
            if any(docs_totals.values()):
                print(f"   📚 Documentation:")
                print(f"      ✅ Added: {docs_totals['added']}")
                print(f"      ✏️  Modified: {docs_totals['modified']}")
                print(f"      ❌ Deleted: {docs_totals['deleted']}")
                total_docs_added += docs_totals['added']
                total_docs_modified += docs_totals['modified']

            # Line statistics
            print(f"   📊 Lines: +{line_changes['lines_added']} -{line_changes['lines_deleted']}")

            total_commits += file_changes['commits']
            total_lines_added += line_changes['lines_added']
            total_lines_deleted += line_changes['lines_deleted']
            total_code_lines_added += line_changes['code_lines_added']
            total_code_lines_deleted += line_changes['code_lines_deleted']
            total_doc_lines_added += line_changes['doc_lines_added']
            total_doc_lines_deleted += line_changes['doc_lines_deleted']
            all_authors.update(file_changes['authors'])

            print()

        # Summary
        print("="*70)
        print("📊 SUMMARY")
        print("="*70)
        print(f"Total Commits: {total_commits}")
        print(f"Active Authors: {len(all_authors)}")
        print(f"\nCode Files:")
        print(f"  ✅ Added: {total_code_added}")
        print(f"  ✏️  Modified: {total_code_modified}")
        print(f"  📊 Total Code Changes: {total_code_added + total_code_modified}")
        print(f"  📈 Code Lines: +{total_code_lines_added} -{total_code_lines_deleted} (net: {total_code_lines_added - total_code_lines_deleted:+d})")
        print(f"\nDocumentation:")
        print(f"  ✅ Added: {total_docs_added}")
        print(f"  ✏️  Modified: {total_docs_modified}")
        print(f"  📊 Total Document Changes: {total_docs_added + total_docs_modified}")
        print(f"  📈 Doc Lines: +{total_doc_lines_added} -{total_doc_lines_deleted} (net: {total_doc_lines_added - total_doc_lines_deleted:+d})")
        print(f"\nTotal Lines: +{total_lines_added} -{total_lines_deleted}")
        print(f"Net Change: {total_lines_added - total_lines_deleted:+d} lines")
        print("="*70 + "\n")

    def save_json(self, results: dict, output_file: str):
        """Save results to JSON file"""
        # Convert sets to lists for JSON serialization
        for repo_data in results['repositories'].values():
            file_changes = repo_data['file_changes']
            if isinstance(file_changes.get('authors'), set):
                file_changes['authors'] = list(file_changes['authors'])

        output_path = Path(output_file)
        with output_path.open('w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"💾 Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Monitor code and documentation changes in Git repositories'
    )
    parser.add_argument(
        '--dir',
        type=str,
        default='/Users/jub/Projects',
        help='Base directory containing repositories (default: /Users/jub/Projects)'
    )
    parser.add_argument(
        '--weeks',
        type=int,
        default=1,
        help='Number of weeks to analyze (default: 1)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file path (optional)'
    )
    parser.add_argument(
        '--author',
        '-a',
        type=str,
        help='Filter commits by author name (case-insensitive substring match)'
    )
    parser.add_argument(
        '--all-branches',
        '-b',
        action='store_true',
        help='Analyze commits across all branches (not just current branch)'
    )

    args = parser.parse_args()

    monitor = CodeMonitor(args.dir, args.weeks, args.author, args.all_branches)

    print(f"🔍 Found {len(monitor.repos)} repositories:")
    for repo in monitor.repos:
        print(f"   - {repo.name}")
    print()

    results = monitor.analyze_repositories()
    monitor.print_report(results)

    if args.output:
        monitor.save_json(results, args.output)


if __name__ == '__main__':
    main()
