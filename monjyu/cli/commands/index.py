# MONJYU CLI - Index Commands
"""
FEAT-008: CLI - index コマンド群
インデックス構築と状態管理
"""

from pathlib import Path
from typing import Optional
import json

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from monjyu.cli.main import OutputFormat, get_monjyu, print_error, print_success

console = Console()
index_app = typer.Typer(help="Index management commands")


@index_app.command("build")
def index_build(
    path: Path = typer.Argument(..., help="Path to documents directory"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
    levels: str = typer.Option(
        "0,1", "--levels", "-l", help="Index levels to build (comma-separated: 0,1)"
    ),
    rebuild: bool = typer.Option(
        False, "--rebuild", "-r", help="Rebuild existing index"
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", "-o", help="Output format"
    ),
):
    """Build index from documents directory"""
    
    # パス検証
    if not path.exists():
        print_error(f"Path not found: {path}")
        raise typer.Exit(1)
    
    if not path.is_dir():
        print_error(f"Path is not a directory: {path}")
        raise typer.Exit(1)
    
    try:
        monjyu = get_monjyu(config)
        
        # プログレスバー表示
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Building index...", total=100)
            
            # インデックス構築
            progress.update(task, completed=20, description="Scanning documents...")
            
            status = monjyu.index(path, show_progress=False)
            
            progress.update(task, completed=100, description="Complete!")
        
        # 結果出力
        if output == OutputFormat.json:
            console.print_json(json.dumps({
                "status": status.index_status.value,
                "documents": status.document_count,
                "text_units": status.text_unit_count,
                "noun_phrases": status.noun_phrase_count,
                "communities": status.community_count,
                "citation_edges": status.citation_edge_count,
            }))
        else:
            console.print(Panel.fit(
                f"[green]✓ Index built successfully[/green]\n\n"
                f"[bold]Statistics:[/bold]\n"
                f"  Documents:      {status.document_count:,}\n"
                f"  Text Units:     {status.text_unit_count:,}\n"
                f"  Noun Phrases:   {status.noun_phrase_count:,}\n"
                f"  Communities:    {status.community_count:,}\n"
                f"  Citation Edges: {status.citation_edge_count:,}",
                title="Index Build Complete",
                border_style="green"
            ))
    
    except FileNotFoundError as e:
        print_error(str(e))
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Index build failed: {e}")
        raise typer.Exit(1)


@index_app.command("status")
def index_status(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", "-o", help="Output format"
    ),
):
    """Show current index status"""
    
    try:
        monjyu = get_monjyu(config)
        status = monjyu.get_status()
        
        if output == OutputFormat.json:
            console.print_json(json.dumps({
                "status": status.index_status.value,
                "is_ready": status.is_ready,
                "levels_built": [l.value for l in status.index_levels_built],
                "documents": status.document_count,
                "text_units": status.text_unit_count,
                "noun_phrases": status.noun_phrase_count,
                "communities": status.community_count,
                "citation_edges": status.citation_edge_count,
                "last_error": status.last_error,
            }))
        else:
            # ステータスに応じた色
            status_color = {
                "not_built": "yellow",
                "building": "blue",
                "ready": "green",
                "error": "red",
            }.get(status.index_status.value, "white")
            
            table = Table(title="MONJYU Index Status")
            table.add_column("Property", style="cyan")
            table.add_column("Value")
            
            table.add_row(
                "Status",
                f"[{status_color}]{status.index_status.value}[/{status_color}]"
            )
            table.add_row(
                "Ready",
                "[green]Yes[/green]" if status.is_ready else "[yellow]No[/yellow]"
            )
            table.add_row(
                "Levels Built",
                str([l.value for l in status.index_levels_built])
            )
            table.add_row("Documents", f"{status.document_count:,}")
            table.add_row("Text Units", f"{status.text_unit_count:,}")
            table.add_row("Noun Phrases", f"{status.noun_phrase_count:,}")
            table.add_row("Communities", f"{status.community_count:,}")
            table.add_row("Citation Edges", f"{status.citation_edge_count:,}")
            
            if status.last_error:
                table.add_row(
                    "Last Error",
                    f"[red]{status.last_error}[/red]"
                )
            
            console.print(table)
    
    except Exception as e:
        print_error(f"Failed to get status: {e}")
        raise typer.Exit(1)


@index_app.command("clear")
def index_clear(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Skip confirmation"
    ),
):
    """Clear the index (remove all indexed data)"""
    
    if not force:
        confirm = typer.confirm("Are you sure you want to clear the index?")
        if not confirm:
            console.print("[dim]Cancelled[/dim]")
            raise typer.Exit(0)
    
    try:
        monjyu = get_monjyu(config)
        
        # ステートをリセット
        from monjyu.api.state import StateManager
        state_manager = StateManager(monjyu.config.output_path)
        state_manager.reset()
        
        print_success("Index cleared successfully")
    
    except Exception as e:
        print_error(f"Failed to clear index: {e}")
        raise typer.Exit(1)
