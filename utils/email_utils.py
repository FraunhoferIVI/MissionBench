"""
Email utilities for sending experiment results and reports.
"""
import smtplib
import json
import subprocess
import os
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, List, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, try loading manually
    def load_env_file():
        env_file = Path(__file__).resolve().parent.parent / ".env"
        if env_file.exists():
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ.setdefault(key.strip(), value.strip())
    load_env_file()


def get_git_commit_info(repo_path: str) -> Dict[str, str]:
    """Get git commit info from a repository.
    
    Args:
        repo_path: Path to the git repository
        
    Returns:
        Dictionary with commit info (hash, branch, message, author)
    """
    try:
        repo_path = Path(repo_path)
        if not repo_path.exists():
            return {"error": f"Repository path not found: {repo_path}"}
        
        # Get commit hash
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            text=True
        ).strip()
        
        # Get branch name
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_path,
            text=True
        ).strip()
        
        # Get commit message
        commit_msg = subprocess.check_output(
            ["git", "log", "-1", "--format=%B"],
            cwd=repo_path,
            text=True
        ).strip()
        
        # Get author
        author = subprocess.check_output(
            ["git", "log", "-1", "--format=%an <%ae>"],
            cwd=repo_path,
            text=True
        ).strip()
        
        return {
            "commit": commit_hash[:8],  # Short hash
            "branch": branch,
            "message": commit_msg.split("\n")[0],  # First line only
            "author": author,
        }
    except Exception as e:
        return {"error": str(e)}


def get_all_repos_commit_info() -> Dict[str, Dict[str, str]]:
    """Get commit info from all relevant repositories.
    
    Returns:
        Dictionary with commit info for each repo
    """
    repos = {
        "MissionBench": Path(__file__).resolve().parent.parent
    }
    
    commit_info = {}
    for repo_name, repo_path in repos.items():
        commit_info[repo_name] = get_git_commit_info(str(repo_path))
    
    return commit_info


def create_email_summary(
    csv_file: Path,
    html_file: Path,
    batch_summary: Dict,
    commit_info: Dict[str, Dict[str, str]],
    pdf_file: Optional[Path] = None,
    tex_file: Optional[Path] = None,
    local_results_directory: Optional[Path] = None,
    results_directory_link: Optional[str] = None,
) -> str:
    """Create an HTML email body with results summary.
    
    Args:
        csv_file: Path to results CSV file
        html_file: Path to results HTML dashboard
        batch_summary: Summary of batch execution
        commit_info: Git commit info for all repos
        pdf_file: Optional path to PDF report
        
    Returns:
        HTML email body as string
    """
    # Build file list
    file_list = f"""
                    <li><strong>Detailed CSV:</strong> {csv_file.name} - All trial results with metadata</li>
                    <li><strong>Statistics CSV:</strong> statistics_summary.csv - Aggregated statistics</li>
                    <li><strong>HTML Dashboard:</strong> {html_file.name} - Interactive web dashboard</li>
    """
    
    if pdf_file and Path(pdf_file).exists():
        file_list += f"""
                    <li><strong>PDF Report:</strong> {Path(pdf_file).name} - Comprehensive analysis with tables and statistics</li>
    """
    elif tex_file and Path(tex_file).exists():
        file_list += f"""
                    <li><strong>LaTeX Source:</strong> {Path(tex_file).name} - PDF was not generated; compile this .tex file manually</li>
    """
    
    local_results_section = ""
    if local_results_directory:
        local_results_section = f"""
            <div class="section">
                <h2>📂 Local Results Directory</h2>
                <p><code>{Path(local_results_directory).resolve()}</code></p>
            </div>
        """

    results_link_section = ""
    if results_directory_link:
        results_link_section = f"""
            <div class="section">
                <h2>🔗 Results Directory Link</h2>
                <p>
                    <a href=\"{results_directory_link}\" target=\"_blank\">{results_directory_link}</a>
                </p>
            </div>
        """

    html_body = f"""
    <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background-color: #2ecc71; color: white; padding: 20px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #3498db; }}
                .summary {{ background-color: #ecf0f1; }}
                .commit-info {{ background-color: #f8f9fa; font-family: monospace; padding: 10px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚁 UAV Mission Planning - Experiment Results</h1>
                <p>Automated experiment execution complete!</p>
            </div>
            
            <div class="section summary">
                <h2>📊 Execution Summary</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Total Experiments</td>
                        <td>{batch_summary.get('total_experiments', 0)}</td>
                    </tr>
                    <tr>
                        <td>Completed</td>
                        <td style="color: green;">{batch_summary.get('successful', batch_summary.get('completed', 0))}</td>
                    </tr>
                    <tr>
                        <td>Failed</td>
                        <td style="color: red;">{batch_summary.get('failed', 0)}</td>
                    </tr>
                    <tr>
                        <td>Pending</td>
                        <td style="color: orange;">{batch_summary.get('pending', 0)}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>📁 Generated Files</h2>
                <ul>
{file_list}
                </ul>
            </div>

            {local_results_section}

            {results_link_section}
            
            <div class="section">
                <h2>🔧 Repository Versions</h2>
    """
    
    for repo_name, info in commit_info.items():
        if "error" not in info:
            html_body += f"""
                <div class="commit-info">
                    <strong>{repo_name}</strong><br>
                    Commit: <code>{info.get('commit', 'N/A')}</code><br>
                    Branch: <code>{info.get('branch', 'N/A')}</code><br>
                    Message: {info.get('message', 'N/A')}<br>
                    Author: {info.get('author', 'N/A')}
                </div>
            """
        else:
            html_body += f"""
                <div class="commit-info">
                    <strong>{repo_name}</strong><br>
                    Error: {info.get('error', 'Unknown error')}
                </div>
            """
    
    html_body += """
            </div>
            
            <div class="section">
                <p style="color: #7f8c8d; font-size: 12px;">
                    This is an automated message from the UAV Mission Planning experiment automation system.
                </p>
            </div>
        </body>
    </html>
    """
    
    return html_body


def send_email(
    to_email: str,
    subject: str,
    html_body: str,
    attachments: Optional[List[Path]] = None,
    gmail_user: Optional[str] = None,
    gmail_password: Optional[str] = None,
) -> bool:
    """Send email with optional attachments using Gmail SMTP.
    
    Args:
        to_email: Recipient email address
        subject: Email subject
        html_body: HTML email body
        attachments: List of file paths to attach
        gmail_user: Gmail address (uses GMAIL_USER env var if not provided)
        gmail_password: Gmail App Password (uses GMAIL_PASSWORD env var if not provided)
        
    Returns:
        True if email sent successfully, False otherwise
    """
    import os
    
    # Get credentials from environment or parameters
    if not gmail_user:
        gmail_user = os.getenv("GMAIL_USER")
    if not gmail_password:
        gmail_password = os.getenv("GMAIL_PASSWORD")
    
    # Clean up password (remove spaces that may have been added for readability)
    if gmail_password:
        gmail_password = gmail_password.replace(" ", "")
    
    if not gmail_user or not gmail_password:
        print("⚠ Email credentials not configured. Set GMAIL_USER and GMAIL_PASSWORD environment variables.")
        print("  For Gmail: Use an App Password (https://myaccount.google.com/apppasswords)")
        return False
    
    try:
        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = gmail_user
        msg["To"] = to_email
        
        # Attach HTML body
        msg.attach(MIMEText(html_body, "html"))
        
        # Attach files
        if attachments:
            for file_path in attachments:
                file_path = Path(file_path)
                if file_path.exists():
                    with open(file_path, "rb") as attachment:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        "Content-Disposition",
                        f"attachment; filename= {file_path.name}",
                    )
                    msg.attach(part)
                    print(f"  ✓ Attached: {file_path.name}")
        
        # Send email via Gmail SMTP
        print(f"📧 Sending email to {to_email}...")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            print(f"   📌 Logging in as {gmail_user}...")
            server.login(gmail_user, gmail_password)
            print(f"   ✓ Logged in successfully")
            server.sendmail(gmail_user, to_email, msg.as_string())
            print(f"   ✓ Email sent to SMTP server")
        
        print(f"✓ Email sent successfully to {to_email}!")
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        print(f"❌ Gmail authentication failed: {e}")
        print(f"   Check that GMAIL_USER and GMAIL_PASSWORD are correct in .env file")
        print(f"   Make sure you're using an App Password, not your regular Google password")
        return False
    except smtplib.SMTPException as e:
        print(f"❌ SMTP error occurred: {e}")
        return False
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        import traceback
        traceback.print_exc()
        return False


def send_experiment_results_email(
    to_email: str,
    csv_file: Path,
    html_file: Path,
    batch_summary: Dict,
    gmail_user: Optional[str] = None,
    gmail_password: Optional[str] = None,
    pdf_file: Optional[Path] = None,
    tex_file: Optional[Path] = None,
    local_results_directory: Optional[Path] = None,
    results_directory_link: Optional[str] = None,
) -> bool:
    """Send experiment results via email.
    
    Args:
        to_email: Recipient email address
        csv_file: Path to results CSV file
        html_file: Path to results HTML dashboard
        batch_summary: Summary of batch execution
        gmail_user: Gmail address (uses GMAIL_USER env var if not provided)
        gmail_password: Gmail App Password (uses GMAIL_PASSWORD env var if not provided)
        pdf_file: Optional path to PDF report file
        
    Returns:
        True if email sent successfully, False otherwise
    """
    # Get commit info
    commit_info = get_all_repos_commit_info()
    
    # Create email body
    html_body = create_email_summary(
        csv_file,
        html_file,
        batch_summary,
        commit_info,
        pdf_file,
        tex_file,
        local_results_directory,
        results_directory_link,
    )
    
    # Prepare attachments list
    attachments = [csv_file, html_file]
    
    # Add statistics summary CSV if it exists
    stats_csv = csv_file.parent / 'statistics_summary.csv'
    if stats_csv.exists():
        attachments.append(stats_csv)
    
    # Add PDF if provided
    if pdf_file and Path(pdf_file).exists():
        attachments.append(pdf_file)
    elif tex_file and Path(tex_file).exists():
        attachments.append(tex_file)
        compile_log = Path(tex_file).with_suffix(".compile.log")
        if compile_log.exists():
            attachments.append(compile_log)
    
    # Send email with attachments
    return send_email(
        to_email=to_email,
        subject="🚁 UAV Mission Planning - Experiment Results",
        html_body=html_body,
        attachments=attachments,
        gmail_user=gmail_user,
        gmail_password=gmail_password,
    )
