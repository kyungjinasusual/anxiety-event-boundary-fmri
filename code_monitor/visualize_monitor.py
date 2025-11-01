#!/usr/bin/env python3
"""
Code Monitor Visualization - Generate charts and HTML reports
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def generate_html_report(json_file: str, output_file: str = None):
    """Generate HTML report from JSON results"""

    with open(json_file, 'r') as f:
        data = json.load(f)

    period = data['period']
    repos = data['repositories']

    # Calculate totals
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

    repo_data = []
    for repo_name, repo_info in repos.items():
        fc = repo_info['file_changes']
        lc = repo_info['line_changes']

        commits = fc.get('commits', 0)
        if commits == 0:
            continue

        code_added = sum(v['added'] for v in fc['code'].values())
        code_modified = sum(v['modified'] for v in fc['code'].values())
        docs_added = sum(v['added'] for v in fc['docs'].values())
        docs_modified = sum(v['modified'] for v in fc['docs'].values())

        total_commits += commits
        total_code_added += code_added
        total_code_modified += code_modified
        total_docs_added += docs_added
        total_docs_modified += docs_modified
        total_lines_added += lc['lines_added']
        total_lines_deleted += lc['lines_deleted']
        total_code_lines_added += lc.get('code_lines_added', 0)
        total_code_lines_deleted += lc.get('code_lines_deleted', 0)
        total_doc_lines_added += lc.get('doc_lines_added', 0)
        total_doc_lines_deleted += lc.get('doc_lines_deleted', 0)
        all_authors.update(fc.get('authors', []))

        repo_data.append({
            'name': repo_name,
            'commits': commits,
            'code_added': code_added,
            'code_modified': code_modified,
            'docs_added': docs_added,
            'docs_modified': docs_modified,
            'lines_added': lc['lines_added'],
            'lines_deleted': lc['lines_deleted'],
            'code_lines_added': lc.get('code_lines_added', 0),
            'code_lines_deleted': lc.get('code_lines_deleted', 0),
            'doc_lines_added': lc.get('doc_lines_added', 0),
            'doc_lines_deleted': lc.get('doc_lines_deleted', 0),
            'authors': fc.get('authors', [])
        })

    # Sort by activity
    repo_data.sort(key=lambda x: x['commits'], reverse=True)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Code Monitor Report - {period['since']} to {period['until']}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .header .period {{
            font-size: 1.2em;
            opacity: 0.9;
        }}

        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }}

        .summary-card {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
        }}

        .summary-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }}

        .summary-card .label {{
            color: #666;
            font-size: 0.95em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .repos {{
            padding: 40px;
        }}

        .repos h2 {{
            font-size: 1.8em;
            margin-bottom: 30px;
            color: #333;
        }}

        .repo-card {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .repo-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}

        .repo-card h3 {{
            font-size: 1.4em;
            margin-bottom: 15px;
            color: #667eea;
            display: flex;
            align-items: center;
        }}

        .repo-card h3::before {{
            content: "📦";
            margin-right: 10px;
        }}

        .repo-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}

        .stat {{
            padding: 12px;
            background: #f8f9fa;
            border-radius: 6px;
            text-align: center;
        }}

        .stat .stat-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }}

        .stat .stat-label {{
            font-size: 0.85em;
            color: #666;
            margin-top: 5px;
        }}

        .authors {{
            margin-top: 15px;
            padding: 12px;
            background: #f0f4ff;
            border-radius: 6px;
            font-size: 0.9em;
            color: #555;
        }}

        .progress-bar {{
            width: 100%;
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            margin-top: 10px;
            overflow: hidden;
        }}

        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
        }}

        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
            background: #f8f9fa;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
            margin-left: 10px;
        }}

        .badge-commits {{
            background: #e3f2fd;
            color: #1976d2;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Code Monitor Report</h1>
            <div class="period">{period['since']} to {period['until']} ({period['weeks']} week{'s' if period['weeks'] > 1 else ''})</div>
        </div>

        <div class="summary">
            <div class="summary-card">
                <div class="label">Total Commits</div>
                <div class="value">{total_commits}</div>
            </div>
            <div class="summary-card">
                <div class="label">Active Authors</div>
                <div class="value">{len(all_authors)}</div>
            </div>
            <div class="summary-card">
                <div class="label">Code Changes</div>
                <div class="value">{total_code_added + total_code_modified}</div>
            </div>
            <div class="summary-card">
                <div class="label">Doc Changes</div>
                <div class="value">{total_docs_added + total_docs_modified}</div>
            </div>
            <div class="summary-card">
                <div class="label">Code Lines</div>
                <div class="value">+{total_code_lines_added:,}</div>
            </div>
            <div class="summary-card">
                <div class="label">Doc Lines</div>
                <div class="value">+{total_doc_lines_added:,}</div>
            </div>
        </div>

        <div class="repos">
            <h2>Repository Activity</h2>
"""

    for repo in repo_data:
        authors_str = ", ".join(repo['authors'][:3])
        if len(repo['authors']) > 3:
            authors_str += f" +{len(repo['authors']) - 3} more"

        net_lines = repo['lines_added'] - repo['lines_deleted']
        net_sign = "+" if net_lines >= 0 else ""

        # Calculate activity percentage for visual bar
        activity_pct = min(100, (repo['commits'] / max(total_commits, 1)) * 100)

        code_net = repo['code_lines_added'] - repo['code_lines_deleted']
        doc_net = repo['doc_lines_added'] - repo['doc_lines_deleted']

        html += f"""
            <div class="repo-card">
                <h3>{repo['name']}<span class="badge badge-commits">{repo['commits']} commits</span></h3>
                <div class="authors">👥 {authors_str}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {activity_pct}%"></div>
                </div>
                <div class="repo-stats">
                    <div class="stat">
                        <div class="stat-value">{repo['code_added']}</div>
                        <div class="stat-label">Code Files Added</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{repo['code_modified']}</div>
                        <div class="stat-label">Code Files Modified</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{repo['docs_added']}</div>
                        <div class="stat-label">Docs Added</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{repo['docs_modified']}</div>
                        <div class="stat-label">Docs Modified</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">+{repo['code_lines_added']:,}</div>
                        <div class="stat-label">Code Lines Added</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{code_net:+,}</div>
                        <div class="stat-label">Code Net Change</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">+{repo['doc_lines_added']:,}</div>
                        <div class="stat-label">Doc Lines Added</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{doc_net:+,}</div>
                        <div class="stat-label">Doc Net Change</div>
                    </div>
                </div>
            </div>
"""

    html += f"""
        </div>

        <div class="footer">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Code Monitor
        </div>
    </div>
</body>
</html>
"""

    # Save HTML file
    if output_file is None:
        output_file = f"code_monitor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

    output_path = Path(output_file)
    output_path.write_text(html)

    print(f"✅ HTML report generated: {output_path.absolute()}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate HTML report from code monitor JSON')
    parser.add_argument('json_file', help='Input JSON file from code_monitor.py')
    parser.add_argument('--output', '-o', help='Output HTML file path')

    args = parser.parse_args()

    if not Path(args.json_file).exists():
        print(f"❌ Error: JSON file not found: {args.json_file}")
        return

    generate_html_report(args.json_file, args.output)


if __name__ == '__main__':
    main()
