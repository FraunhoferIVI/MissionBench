"""
LaTeX Report Generator: Creates professional experiment reports with statistics.

Generates LaTeX documents with:
- Experiment metadata (git commits, configs, timestamps)
- Overall statistics (success rate, std dev)
- Mission-wise breakdown
- Ready for Overleaf and paper inclusion
"""

import json
import subprocess
import smtplib
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import statistics


class LaTeXReporter:
    """
    Generates LaTeX reports with experiment statistics.
    
    Responsibilities:
    - Compute overall and mission-wise statistics with std dev
    - Extract git commit ID from MissionBench repository
    - Generate LaTeX tables and documents
    - Compile to PDF
    - Send via email
    """
    
    def __init__(self, results: List[Dict[str, Any]], results_base_dir: Path, config: Dict[str, Any] = None):
        """
        Args:
            results: List of experiment results
            results_base_dir: Base directory containing experiment results
            config: Optional configuration for report generation
        """
        self.results = results
        self.results_base_dir = Path(results_base_dir)
        self.config = config or {}
        self.git_commits = {}
        self.overall_stats = {}
        self.mission_stats = {}
        
    def _get_git_commit(self, repo_path: str) -> Optional[str]:
        """
        Get the current git commit hash from a repository.
        
        Args:
            repo_path: Path to git repository
            
        Returns:
            Commit hash (short) or None if not a git repo
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            print(f"  ⚠ Failed to get git commit for {repo_path}: {e}")
        return None
    
    def _get_git_branch(self, repo_path: str) -> Optional[str]:
        """Get the current git branch name."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def extract_git_info(self, repo_paths: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, str]]:
        """
        Extract git information from MissionBench repository.
        
        Args:
            repo_paths: Optional dict mapping repo names to paths.
                If omitted, defaults to current MissionBench repository.
            
        Returns:
            Dict with git info for each repo
        """
        _ = repo_paths  # kept for backward compatibility; single-repo mode ignores external repo maps

        print("\nExtracting git commit information...")
        git_info = {}

        # MissionBench now runs as a single-repo setup.
        # Keep report metadata deterministic by always using only this repo.
        repo_name = "MissionBench"
        repo_path = Path(__file__).resolve().parent.parent

        if repo_path.exists():
            commit = self._get_git_commit(str(repo_path))
            branch = self._get_git_branch(str(repo_path))
            git_info[repo_name] = {
                "commit": commit or "unknown",
                "branch": branch or "unknown",
                "path": str(repo_path)
            }
            print(f"  {repo_name}: {commit or 'N/A'} ({branch or 'N/A'})")
        else:
            print(f"  ⚠ {repo_name}: Path not found - {repo_path}")
            git_info[repo_name] = {"commit": "N/A", "branch": "N/A", "path": str(repo_path)}
        
        self.git_commits = git_info
        return git_info
    
    def compute_overall_statistics(self) -> Dict[str, Any]:
        """
        Compute overall statistics across all experiments.
        Uses mission-level means for standard deviation calculation.
        
        Returns:
            Dict with overall success rate, std dev, time stats, etc.
        """
        if not self.results:
            return {}
        
        # Group results by mission to compute mission-level means
        by_mission = {}
        for result in self.results:
            mission = result.get("mission_result", {}) or {}
            scenario = mission.get("scenario_name", "unknown")
                # MissionBench now runs as a single-repo setup.
            if scenario not in by_mission:
                by_mission[scenario] = {
                    "success_values": [],
                    "osr_5_values": [],
                    "osr_10_values": [],
                    "osr_20_values": [],
                    "sr5_values": [],
                    "sr10_values": [],
                    "time_values": [],
                }
            
            by_mission[scenario]["success_values"].append(1 if result.get("success") else 0)
            by_mission[scenario]["sr5_values"].append(mission.get("sr5", 0) if mission.get("sr5") is not None else None)
            by_mission[scenario]["sr10_values"].append(mission.get("sr10", 0) if mission.get("sr10") is not None else None)
            if mission.get("OSR_5") is not None:
                by_mission[scenario]["osr_5_values"].append(mission["OSR_5"])
            if mission.get("OSR_10") is not None:
                by_mission[scenario]["osr_10_values"].append(mission["OSR_10"])
            if mission.get("OSR_20") is not None:
                by_mission[scenario]["osr_20_values"].append(mission["OSR_20"])
            if result.get("total_time"):
                by_mission[scenario]["time_values"].append(result["total_time"])
        
        # Calculate mission-level means, then std dev of those means
        mission_success_rates = []
        mission_sr5_rates = []
        mission_sr10_rates = []
        mission_osr5_rates = []
        mission_osr10_rates = []
        mission_osr20_rates = []
        
        for scenario, data in by_mission.items():
            if data["success_values"]:
                mission_success_rates.append(statistics.mean(data["success_values"]) * 100)
            sr5_vals = [v for v in data["sr5_values"] if v is not None]
            if sr5_vals:
                mission_sr5_rates.append(statistics.mean(sr5_vals) * 100)
            sr10_vals = [v for v in data["sr10_values"] if v is not None]
            if sr10_vals:
                mission_sr10_rates.append(statistics.mean(sr10_vals) * 100)
            if data["osr_5_values"]:
                mission_osr5_rates.append(statistics.mean(data["osr_5_values"]) * 100)
            if data["osr_10_values"]:
                mission_osr10_rates.append(statistics.mean(data["osr_10_values"]) * 100)
            if data["osr_20_values"]:
                mission_osr20_rates.append(statistics.mean(data["osr_20_values"]) * 100)
        
        # Overall rates (pooled across all trials)
        success_values = [1 if r.get("success") else 0 for r in self.results]
        sr5_values = [r.get("mission_result", {}).get("sr5") for r in self.results 
                      if r.get("mission_result", {}).get("sr5") is not None]
        sr10_values = [r.get("mission_result", {}).get("sr10") for r in self.results 
                       if r.get("mission_result", {}).get("sr10") is not None]
        osr_5_values = [r.get("mission_result", {}).get("OSR_5", 0) for r in self.results 
                        if r.get("mission_result", {}).get("OSR_5") is not None]
        osr_10_values = [r.get("mission_result", {}).get("OSR_10", 0) for r in self.results 
                         if r.get("mission_result", {}).get("OSR_10") is not None]
        osr_20_values = [r.get("mission_result", {}).get("OSR_20", 0) for r in self.results 
                         if r.get("mission_result", {}).get("OSR_20") is not None]
        
        # Time statistics
        time_values = [r.get("total_time", 0) for r in self.results if r.get("total_time")]
        
        stats = {
            "total_experiments": len(self.results),
            "total_successful": sum(success_values),
            "total_failed": len(success_values) - sum(success_values),
            "success_rate": statistics.mean(success_values) * 100,
            "success_std": statistics.stdev(mission_success_rates) if len(mission_success_rates) > 1 else 0,
            "sr5_rate": statistics.mean(sr5_values) * 100 if sr5_values else 0,
            "sr5_std": statistics.stdev(mission_sr5_rates) if len(mission_sr5_rates) > 1 else 0,
            "sr10_rate": statistics.mean(sr10_values) * 100 if sr10_values else 0,
            "sr10_std": statistics.stdev(mission_sr10_rates) if len(mission_sr10_rates) > 1 else 0,
            "osr_5_rate": statistics.mean(osr_5_values) * 100 if osr_5_values else 0,
            "osr_5_std": statistics.stdev(mission_osr5_rates) if len(mission_osr5_rates) > 1 else 0,
            "osr_10_rate": statistics.mean(osr_10_values) * 100 if osr_10_values else 0,
            "osr_10_std": statistics.stdev(mission_osr10_rates) if len(mission_osr10_rates) > 1 else 0,
            "osr_20_rate": statistics.mean(osr_20_values) * 100 if osr_20_values else 0,
            "osr_20_std": statistics.stdev(mission_osr20_rates) if len(mission_osr20_rates) > 1 else 0,
            "avg_time": statistics.mean(time_values) if time_values else 0,
            "std_time": statistics.stdev(time_values) if len(time_values) > 1 else 0,
            "min_time": min(time_values) if time_values else 0,
            "max_time": max(time_values) if time_values else 0,
        }
        
        self.overall_stats = stats
        return stats
    
    def compute_mission_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Compute statistics grouped by mission scenario.
        
        Returns:
            Dict mapping mission names to their statistics
        """
        if not self.results:
            return {}
        
        # Group by mission
        by_mission = {}
        
        for result in self.results:
            mission = result.get("mission_result", {}) or {}
            scenario = mission.get("scenario_name", "unknown")
            
            if scenario not in by_mission:
                by_mission[scenario] = {
                    "trials": [],
                    "success_values": [],
                    "osr_5_values": [],
                    "osr_10_values": [],
                    "osr_20_values": [],
                    "time_values": [],
                    "distances": [],
                }
            
            by_mission[scenario]["trials"].append(result)
            by_mission[scenario]["success_values"].append(1 if result.get("success") else 0)
            
            if mission.get("OSR_5") is not None:
                by_mission[scenario]["osr_5_values"].append(mission["OSR_5"])
            if mission.get("OSR_10") is not None:
                by_mission[scenario]["osr_10_values"].append(mission["OSR_10"])
            if mission.get("OSR_20") is not None:
                by_mission[scenario]["osr_20_values"].append(mission["OSR_20"])
            if result.get("total_time"):
                by_mission[scenario]["time_values"].append(result["total_time"])
            if mission.get("min_distance_to_target") is not None:
                by_mission[scenario]["distances"].append(mission["min_distance_to_target"])
        
        # Compute statistics for each mission
        mission_stats = {}
        for scenario, data in by_mission.items():
            success_vals = data["success_values"]
            osr_5 = data["osr_5_values"]
            osr_10 = data["osr_10_values"]
            osr_20 = data["osr_20_values"]
            times = data["time_values"]
            distances = data["distances"]
            
            mission_stats[scenario] = {
                "total_trials": len(success_vals),
                "successful_trials": sum(success_vals),
                "success_rate": statistics.mean(success_vals) * 100 if success_vals else 0,
                "success_std": statistics.stdev(success_vals) * 100 if len(success_vals) > 1 else 0,
                "osr_5_rate": statistics.mean(osr_5) * 100 if osr_5 else None,
                "osr_5_std": statistics.stdev(osr_5) * 100 if len(osr_5) > 1 else 0,
                "osr_10_rate": statistics.mean(osr_10) * 100 if osr_10 else None,
                "osr_10_std": statistics.stdev(osr_10) * 100 if len(osr_10) > 1 else 0,
                "osr_20_rate": statistics.mean(osr_20) * 100 if osr_20 else None,
                "osr_20_std": statistics.stdev(osr_20) * 100 if len(osr_20) > 1 else 0,
                "avg_time": statistics.mean(times) if times else None,
                "std_time": statistics.stdev(times) if len(times) > 1 else 0,
                "avg_distance": statistics.mean(distances) if distances else None,
                "std_distance": statistics.stdev(distances) if len(distances) > 1 else 0,
            }
        
        self.mission_stats = mission_stats
        return mission_stats
    
    def generate_latex(self, output_file: Path = None, title: str = "Experiment Results") -> Path:
        """
        Generate LaTeX document with experiment statistics.
        
        Args:
            output_file: Output .tex file path
            title: Report title
            
        Returns:
            Path to generated .tex file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.results_base_dir / f"report_{timestamp}.tex"
        
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure we have stats
        if not self.overall_stats:
            self.compute_overall_statistics()
        if not self.mission_stats:
            self.compute_mission_statistics()
        
        print(f"\nGenerating LaTeX report: {output_file}")
        
        latex_content = self._generate_latex_content(title)
        
        with open(output_file, "w") as f:
            f.write(latex_content)
        
        print(f"  ✓ LaTeX generated: {output_file}")
        return output_file
    
    def _generate_latex_content(self, title: str) -> str:
        """Generate the full LaTeX document content."""
        
        # Get experiment metadata
        first_result = self.results[0] if self.results else {}
        config = first_result.get("config", {})
        
        # Format git info table
        git_table = self._generate_git_table()
        
        # Format config info
        config_info = self._generate_config_section(config)
        
        # Overall statistics table
        overall_table = self._generate_overall_stats_table()
        
        # Mission-wise statistics table
        mission_table = self._generate_mission_stats_table()
        
        latex = f"""\\documentclass[11pt,a4paper]{{article}}
\\usepackage{{booktabs}}
\\usepackage{{geometry}}
\\usepackage{{multirow}}
\\usepackage{{array}}
\\usepackage{{longtable}}
\\usepackage{{xcolor}}
\\usepackage{{hyperref}}
\\usepackage{{float}}

\\geometry{{margin=2.5cm}}

\\title{{{title}}}
\\author{{Automated Experiment Report}}
\\date{{{datetime.now().strftime("%B %d, %Y - %H:%M:%S")}}}

\\begin{{document}}

\\maketitle

\\section{{Experiment Metadata}}

\\subsection{{Repository Information}}
{git_table}

\\subsection{{Configuration}}
{config_info}

\\section{{Overall Statistics}}

{overall_table}

\\section{{Mission-wise Statistics}}

{mission_table}

\\section{{Summary}}

This report was automatically generated from \\textbf{{{self.overall_stats.get('total_experiments', 0)}}} experimental trials.
The overall success rate is \\textbf{{{self.overall_stats.get('success_rate', 0):.2f}\\%}} 
with a standard deviation of \\textbf{{{self.overall_stats.get('success_std', 0):.2f}\\%}} across all missions.

\\end{{document}}
"""
        return latex
    
    def _generate_git_table(self) -> str:
        """Generate LaTeX table for git commit information."""
        if not self.git_commits:
            return "\\textit{No git information available.}\n"

        # Final report should include only the current MissionBench repo commit metadata.
        if "MissionBench" in self.git_commits:
            git_entries = {"MissionBench": self.git_commits["MissionBench"]}
        else:
            first_repo_name = next(iter(self.git_commits))
            git_entries = {first_repo_name: self.git_commits[first_repo_name]}

        rows = []
            for repo_name, info in git_entries.items():
            # Handle both dict and string formats
            if isinstance(info, dict):
                commit = info.get("commit", "N/A")
                branch = info.get("branch", "N/A")
                message = info.get("message", "N/A")
                # Escape special characters
                repo_name = repo_name.replace("_", "\\_")
                branch = branch.replace("_", "\\_")
                message = message.replace("&", "\\&").replace("_", "\\_").replace("#", "\\#").replace("%", "\\%")
                # Truncate long messages
                if len(message) > 60:
                    message = message[:57] + "..."
            else:
                commit = str(info)
                branch = "N/A"
                message = "N/A"
                repo_name = repo_name.replace("_", "\\_")
            # Use smaller font for message to prevent overflow
            rows.append(f"{repo_name} & \\texttt{{{commit}}} & {branch} & {{\\small {message}}} \\\\")
        
        table = f"""\\begin{{table}}[H]
    \\centering
    \\small
    \\begin{{tabular}}{{lllp{{6cm}}}}
    \\toprule
    \\textbf{{Repository}} & \\textbf{{Commit}} & \\textbf{{Branch}} & \\textbf{{Message}} \\\\
    \\midrule
    {chr(10).join(rows)}
    \\bottomrule
    \\end{{tabular}}
        \\caption{{Git commit information for MissionBench repository.}}
    \\label{{tab:git-info}}
    \\end{{table}}
    """
        return table
    
    def _generate_config_section(self, config: Dict[str, Any]) -> str:
        """Generate configuration information section from configs/experiment_configs.yaml file."""
        # Try to load configs.yaml from common MissionBench locations
        config_yaml_path = Path("configs/experiment_configs.yaml")
        if not config_yaml_path.exists():
            config_yaml_path = Path(__file__).resolve().parent.parent / "configs" / "experiment_configs.yaml"
        if not config_yaml_path.exists():
            config_yaml_path = self.results_base_dir.parent / "configs" / "experiment_configs.yaml"
        
        config_content = ""
        if config_yaml_path.exists():
            try:
                with open(config_yaml_path, 'r') as f:
                    config_content = f.read()
            except Exception:
                pass
        
        if not config_content:
            return "\\textit{Configuration file (configs/experiment_configs.yaml) not found.}\n"
        
        # No escaping needed - verbatim mode handles all special characters naturally
        return f"""\\begin{{verbatim}}
{config_content}
\\end{{verbatim}}
"""
    
    def _generate_config_section_old(self, config: Dict[str, Any]) -> str:
        """Generate configuration information section (OLD - from experiment config)."""
        if not config:
            return "\\textit{No configuration available.}\n"
        
        # Key config parameters
        important_keys = [
            "model_name", "scenario_name", "environment_name",
            "max_actions", "max_step_size", "strategy"
        ]
        
        def escape_latex(text: str) -> str:
            """Escape special LaTeX characters."""
            text = str(text)
            # Escape special characters
            text = text.replace("\\", "\\textbackslash{}")
            text = text.replace("&", "\\&")
            text = text.replace("%", "\\%")
            text = text.replace("$", "\\$")
            text = text.replace("#", "\\#")
            text = text.replace("_", "\\_")
            text = text.replace("{", "\\{")
            text = text.replace("}", "\\}")
            text = text.replace("~", "\\textasciitilde{}")
            text = text.replace("^", "\\textasciicircum{}")
            text = text.replace("[", "{[}")
            text = text.replace("]", "{]}")
            return text
        
        items = []
        for key in important_keys:
            if key in config:
                value = escape_latex(config[key])
                items.append(f"\\item \\textbf{{{key.replace('_', ' ').title()}}}: {value}")
        
        # Add any other config items not in important_keys  
        for key, value in config.items():
            if key not in important_keys:
                value_str = str(value)
                if len(value_str) > 50:  # Truncate long values
                    value_str = value_str[:50] + "..."
                value_str = escape_latex(value_str)
                items.append(f"\\item \\textbf{{{key.replace('_', ' ').title()}}}: {value_str}")
        
        if items:
            return "\\begin{itemize}\n" + "\n".join(items) + "\n\\end{itemize}\n"
        else:
            return "\\textit{No configuration parameters recorded.}\n"
    
    def _generate_overall_stats_table(self) -> str:
        """Generate LaTeX table for overall statistics."""
        stats = self.overall_stats
        
        table = f"""\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{lr}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\midrule
Total Experiments & {stats.get('total_experiments', 0)} \\\\
Successful & {stats.get('total_successful', 0)} \\\\
Failed & {stats.get('total_failed', 0)} \\\\
\\midrule
Success Rate & ${stats.get('success_rate', 0):.2f} \\pm {stats.get('success_std', 0):.2f}$\\% \\\\
SR@5m & ${stats.get('sr5_rate', 0):.2f} \\pm {stats.get('sr5_std', 0):.2f}$\\% \\\\
SR@10m & ${stats.get('sr10_rate', 0):.2f} \\pm {stats.get('sr10_std', 0):.2f}$\\% \\\\
OSR@5m & ${stats.get('osr_5_rate', 0):.2f} \\pm {stats.get('osr_5_std', 0):.2f}$\\% \\\\
OSR@10m & ${stats.get('osr_10_rate', 0):.2f} \\pm {stats.get('osr_10_std', 0):.2f}$\\% \\\\
OSR@20m & ${stats.get('osr_20_rate', 0):.2f} \\pm {stats.get('osr_20_std', 0):.2f}$\\% \\\\
\\midrule
Avg. Time (min) & ${stats.get('avg_time', 0)/60:.2f} \\pm {stats.get('std_time', 0)/60:.2f}$ \\\\
Min Time (min) & {stats.get('min_time', 0)/60:.2f} \\\\
Max Time (min) & {stats.get('max_time', 0)/60:.2f} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Overall experiment statistics across all missions and trials.}}
\\label{{tab:overall-stats}}
\\end{{table}}
"""
        return table
    
    def _generate_mission_stats_table(self) -> str:
        """Generate LaTeX table for mission-wise statistics."""
        if not self.mission_stats:
            return "\\textit{No mission-wise statistics available.}\n"
        
        # Sort missions by name
        sorted_missions = sorted(self.mission_stats.items())
        
        rows = []
        for mission_name, stats in sorted_missions:
            # Escape underscores and other special characters in mission name
            display_name = mission_name.replace("_", "\\_").replace("&", "\\&").replace("#", "\\#")
            
            # Format success rate with std
            success_str = f"${stats['success_rate']:.1f} \\pm {stats['success_std']:.1f}$"
            
            # Format OSR (use OSR@10 as primary)
            if stats['osr_10_rate'] is not None:
                osr_str = f"${stats['osr_10_rate']:.1f} \\pm {stats['osr_10_std']:.1f}$"
            else:
                osr_str = "N/A"
            
            # Format time (convert to minutes)
            if stats['avg_time'] is not None:
                time_str = f"${stats['avg_time']/60:.1f} \\pm {stats['std_time']/60:.1f}$"
            else:
                time_str = "N/A"
            
            rows.append(
                f"{display_name} & {stats['total_trials']} & "
                f"{success_str} & {osr_str} & {time_str} \\\\"
            )
        
        table = f"""\\begin{{longtable}}{{lrccr}}
\\caption{{Mission-wise statistics showing success rate and standard deviation across trials.}}
\\label{{tab:mission-stats}}\\\\
\\toprule
\\textbf{{Mission}} & \\textbf{{Trials}} & \\textbf{{Success Rate (\\%)}} & \\textbf{{OSR@10m (\\%)}} & \\textbf{{Avg. Time (min)}} \\\\
\\midrule
\\endfirsthead

\\multicolumn{{5}}{{c}}{{\\tablename\\ \\thetable\\ -- continued from previous page}} \\\\
\\toprule
\\textbf{{Mission}} & \\textbf{{Trials}} & \\textbf{{Success Rate (\\%)}} & \\textbf{{OSR@10m (\\%)}} & \\textbf{{Avg. Time (min)}} \\\\
\\midrule
\\endhead

\\midrule
\\multicolumn{{5}}{{r}}{{Continued on next page}} \\\\
\\endfoot

\\bottomrule
\\endlastfoot

{chr(10).join(rows)}
\\end{{longtable}}
"""
        return table
    
    def compile_pdf(self, tex_file: Path, output_pdf: Path = None) -> Optional[Path]:
        """
        Compile LaTeX file to PDF using pdflatex.
        
        Args:
            tex_file: Path to .tex file
            output_pdf: Optional output PDF path (default: same name as tex)
            
        Returns:
            Path to generated PDF or None if compilation failed
        """
        tex_file = Path(tex_file)
        
        if output_pdf is None:
            output_pdf = tex_file.with_suffix(".pdf")
        else:
            output_pdf = Path(output_pdf)
        
        # Ensure output_pdf is absolute
        if not output_pdf.is_absolute():
            output_pdf = tex_file.parent / output_pdf.name
        
        print(f"\nCompiling PDF from {tex_file.name}...")
        compile_log_file = tex_file.with_suffix(".compile.log")
        
        try:
            if shutil.which("pdflatex") is None:
                print("  ⚠ pdflatex not found in PATH. Install TeX Live or MiKTeX to compile PDFs.")
                print(f"  ⚠ Saved LaTeX source for manual compilation: {tex_file}")
                return None

            run_logs = []
            # Run pdflatex twice for proper references
            for run in [1, 2]:
                result = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", str(tex_file.name)],
                    cwd=str(tex_file.parent.absolute()),
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                combined_log = (
                    f"===== pdflatex run {run} =====\n"
                    f"returncode={result.returncode}\n\n"
                    f"--- stdout ---\n{result.stdout}\n"
                    f"--- stderr ---\n{result.stderr}\n"
                )
                run_logs.append(combined_log)
                
                if result.returncode != 0:
                    print(f"  ⚠ pdflatex run {run} returned error code (may still succeed)")
                else:
                    print(f"  ✓ Run {run} complete")

            compile_log_file.write_text("\n".join(run_logs), encoding="utf-8")
            
            # Check if PDF was created regardless of return code
            # (pdflatex can return non-zero for warnings but still create PDF)
            if output_pdf.exists():
                print(f"  ✓ PDF generated: {output_pdf}")
                print(f"  ✓ Compile log: {compile_log_file}")
                
                # Clean up auxiliary files
                for ext in [".aux", ".log", ".out"]:
                    aux_file = tex_file.with_suffix(ext)
                    if aux_file.exists():
                        aux_file.unlink()
                
                return output_pdf
            else:
                print(f"  ⚠ PDF file not found after compilation")
                print(f"  ⚠ Check compile log: {compile_log_file}")
                return None
                
        except FileNotFoundError:
            print("  ⚠ pdflatex not found. Install TeX Live or MiKTeX to compile PDFs.")
            print("  The .tex file has been generated and can be compiled manually or in Overleaf.")
            return None
        except Exception as e:
            print(f"  ⚠ PDF compilation failed: {e}")
            print(f"  ⚠ LaTeX source retained at: {tex_file}")
            return None
    
    def send_email(
        self,
        pdf_file: Path,
        recipient: str,
        subject: str = None,
        smtp_config: Dict[str, Any] = None
    ) -> bool:
        """
        Send PDF report via email.
        
        Args:
            pdf_file: Path to PDF file to send
            recipient: Email address of recipient
            subject: Email subject (auto-generated if None)
            smtp_config: SMTP configuration dict with keys:
                - server: SMTP server address
                - port: SMTP port (default 587)
                - username: SMTP username
                - password: SMTP password
                - sender: Sender email address
                
        Returns:
            True if email sent successfully, False otherwise
        """
        if not smtp_config:
            print("  ⚠ No SMTP configuration provided. Skipping email.")
            return False
        
        if not Path(pdf_file).exists():
            print(f"  ⚠ PDF file not found: {pdf_file}")
            return False
        
        # Default subject
        if subject is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            subject = f"Experiment Report - {timestamp}"
        
        print(f"\nSending email to {recipient}...")
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = smtp_config.get('sender', smtp_config['username'])
            msg['To'] = recipient
            msg['Subject'] = subject
            
            # Email body
            body = f"""
Automated Experiment Report

This email contains the experiment report generated on {datetime.now().strftime("%Y-%m-%d at %H:%M:%S")}.

Overall Statistics:
- Total Experiments: {self.overall_stats.get('total_experiments', 0)}
- Success Rate: {self.overall_stats.get('success_rate', 0):.2f}% ± {self.overall_stats.get('success_std', 0):.2f}%
- Average Time: {self.overall_stats.get('avg_time', 0):.2f}s ± {self.overall_stats.get('std_time', 0):.2f}s

Please see the attached PDF for detailed results.

---
This is an automated message from the experiment reporting system.
"""
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach PDF
            with open(pdf_file, 'rb') as f:
                pdf_attachment = MIMEApplication(f.read(), _subtype='pdf')
                pdf_attachment.add_header(
                    'Content-Disposition', 'attachment',
                    filename=Path(pdf_file).name
                )
                msg.attach(pdf_attachment)
            
            # Send email
            server = smtplib.SMTP(
                smtp_config.get('server', 'smtp.gmail.com'),
                smtp_config.get('port', 587)
            )
            server.starttls()
            server.login(smtp_config['username'], smtp_config['password'])
            server.send_message(msg)
            server.quit()
            
            print(f"  ✓ Email sent successfully to {recipient}")
            return True
            
        except Exception as e:
            print(f"  ⚠ Failed to send email: {e}")
            return False


def main():
    """
    Standalone script to generate LaTeX report from experiment results.
    
    Usage:
        python latex_reporter.py [results_directory]
    """
    import sys
    
    # Get results directory from command line or use default
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = Path("data/results")
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        sys.exit(1)
    
    print(f"📂 Loading results from: {results_dir}")
    
    # Load results from result.json files
    results = []
    result_files = list(results_dir.glob("*/result.json"))
    
    for result_file in result_files:
        try:
            with open(result_file, "r") as f:
                result = json.load(f)
                if result.get("disable_from_dashboard", 0) != 1:
                    results.append(result)
        except Exception as e:
            print(f"  ⚠ Failed to load {result_file}: {e}")
    
    if not results:
        print("❌ No results found")
        sys.exit(1)
    
    print(f"✓ Loaded {len(results)} results")
    
    # Initialize reporter
    reporter = LaTeXReporter(results, results_dir)
    
    # Extract git information for current MissionBench repository only
    reporter.extract_git_info()
    
    # Compute statistics
    reporter.compute_overall_statistics()
    reporter.compute_mission_statistics()
    
    # Generate LaTeX
    tex_file = reporter.generate_latex(title="MissionBench Experiment Results")
    
    # Compile PDF
    pdf_file = reporter.compile_pdf(tex_file)
    
    # Optionally send email (requires SMTP config)
    # smtp_config = {
    #     'server': 'smtp.gmail.com',
    #     'port': 587,
    #     'username': 'your_email@gmail.com',
    #     'password': 'your_app_password',
    #     'sender': 'your_email@gmail.com'
    # }
    # reporter.send_email(pdf_file, 'recipient@example.com', smtp_config=smtp_config)
    
    print("\n" + "="*70)
    print("✓ Report generation complete!")
    print("="*70)
    print(f"  LaTeX: {tex_file}")
    if pdf_file:
        print(f"  PDF: {pdf_file}")
    print("="*70)


if __name__ == "__main__":
    main()
