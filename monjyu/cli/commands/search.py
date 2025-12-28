# MONJYU CLI - Search Commands
"""
FEAT-008: CLI - search コマンド群
検索クエリ実行と対話モード
"""

from pathlib import Path
from typing import Optional
import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

from monjyu.cli.main import OutputFormat, get_monjyu, print_error
from monjyu.api import SearchMode

console = Console()
search_app = typer.Typer(help="Search commands")


@search_app.callback(invoke_without_command=True)
def search_default(
    ctx: typer.Context,
    query: Optional[str] = typer.Argument(None, help="Search query"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
    mode: str = typer.Option(
        "lazy", "--mode", "-m",
        help="Search mode: vector, lazy, auto"
    ),
    top_k: int = typer.Option(
        10, "--top-k", "-k", help="Number of results"
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", "-o", help="Output format"
    ),
):
    """Execute search query (default command)"""
    
    # サブコマンドが呼ばれた場合はスキップ
    if ctx.invoked_subcommand is not None:
        return
    
    if query is None:
        console.print("[yellow]Please provide a search query[/yellow]")
        console.print("\nUsage: monjyu search \"your query here\"")
        console.print("       monjyu search interactive")
        raise typer.Exit(1)
    
    try:
        monjyu = get_monjyu(config)
        
        # 検索モード変換
        try:
            search_mode = SearchMode(mode)
        except ValueError:
            print_error(f"Invalid search mode: {mode}")
            console.print("Valid modes: vector, lazy, auto")
            raise typer.Exit(1)
        
        # 検索実行
        with console.status("[bold green]Searching...", spinner="dots"):
            result = monjyu.search(query, mode=search_mode, top_k=top_k)
        
        # 結果出力
        if output == OutputFormat.json:
            console.print_json(json.dumps({
                "query": result.query,
                "answer": result.answer,
                "citations": [
                    {
                        "doc_id": c.doc_id,
                        "title": c.title,
                        "text": c.text,
                        "relevance_score": c.relevance_score,
                    }
                    for c in result.citations
                ],
                "search_mode": result.search_mode.value,
                "search_level": result.search_level,
                "total_time_ms": result.total_time_ms,
                "llm_calls": result.llm_calls,
            }))
        else:
            # 回答パネル
            console.print(Panel(
                Markdown(result.answer),
                title=f"[bold]Answer[/bold] "
                      f"[dim](mode: {result.search_mode.value}, "
                      f"level: {result.search_level})[/dim]",
                border_style="green",
            ))
            
            # 引用表示
            if result.citations:
                console.print("\n[bold]Citations:[/bold]")
                for i, citation in enumerate(result.citations[:5], 1):
                    title = citation.title or citation.doc_id
                    text_preview = citation.text[:100] if citation.text else ""
                    if len(citation.text or "") > 100:
                        text_preview += "..."
                    
                    console.print(f"  [{i}] [cyan]{title}[/cyan]")
                    if text_preview:
                        console.print(f"      [dim]{text_preview}[/dim]")
            
            # メタデータ
            console.print(
                f"\n[dim]Time: {result.total_time_ms:.0f}ms | "
                f"LLM calls: {result.llm_calls}[/dim]"
            )
    
    except Exception as e:
        print_error(f"Search failed: {e}")
        raise typer.Exit(1)


@search_app.command("interactive")
def search_interactive(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
    mode: str = typer.Option(
        "lazy", "--mode", "-m", help="Initial search mode"
    ),
):
    """Start interactive search mode"""
    
    try:
        monjyu = get_monjyu(config)
        
        # 初期モード
        try:
            current_mode = SearchMode(mode)
        except ValueError:
            current_mode = SearchMode.LAZY
        
        # ウェルカムメッセージ
        console.print(Panel.fit(
            "[bold cyan]MONJYU Interactive Search[/bold cyan]\n\n"
            f"Current mode: [green]{current_mode.value}[/green]\n\n"
            "[dim]Commands:[/dim]\n"
            "  [cyan]exit[/cyan] or [cyan]quit[/cyan] - Exit interactive mode\n"
            "  [cyan]mode <vector|lazy|auto>[/cyan] - Change search mode\n"
            "  [cyan]clear[/cyan] - Clear screen\n"
            "  [cyan]help[/cyan] - Show this message",
            border_style="blue",
        ))
        
        # 対話ループ
        while True:
            try:
                query = console.input("\n[bold cyan]Query>[/bold cyan] ")
                query = query.strip()
                
                # 終了コマンド
                if query.lower() in ["exit", "quit", "q"]:
                    console.print("[dim]Goodbye![/dim]")
                    break
                
                # モード変更
                if query.lower().startswith("mode "):
                    new_mode = query.split(" ", 1)[1].strip()
                    try:
                        current_mode = SearchMode(new_mode)
                        console.print(
                            f"[green]✓ Mode changed to: {current_mode.value}[/green]"
                        )
                    except ValueError:
                        console.print(f"[red]Invalid mode: {new_mode}[/red]")
                        console.print("Valid modes: vector, lazy, auto")
                    continue
                
                # 画面クリア
                if query.lower() == "clear":
                    console.clear()
                    continue
                
                # ヘルプ
                if query.lower() == "help":
                    console.print(
                        "\n[bold]Commands:[/bold]\n"
                        "  exit, quit, q - Exit\n"
                        "  mode <mode> - Change mode (vector, lazy, auto)\n"
                        "  clear - Clear screen\n"
                        "  help - Show this message"
                    )
                    continue
                
                # 空クエリ
                if not query:
                    continue
                
                # 検索実行
                with console.status("[bold green]Searching...", spinner="dots"):
                    result = monjyu.search(query, mode=current_mode)
                
                # 結果表示
                console.print(Panel(
                    Markdown(result.answer),
                    title=f"Answer [dim]({result.search_mode.value})[/dim]",
                    border_style="green",
                ))
                
                if result.citations:
                    console.print("[dim]Citations:[/dim]")
                    for i, c in enumerate(result.citations[:3], 1):
                        title = c.title or c.doc_id
                        console.print(f"  [{i}] {title}")
                
                console.print(
                    f"[dim]Time: {result.total_time_ms:.0f}ms[/dim]"
                )
            
            except KeyboardInterrupt:
                console.print("\n[dim]Ctrl+C pressed. Type 'exit' to quit.[/dim]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    except Exception as e:
        print_error(f"Failed to start interactive mode: {e}")
        raise typer.Exit(1)


@search_app.command("history")
def search_history(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
    limit: int = typer.Option(
        10, "--limit", "-n", help="Number of recent searches"
    ),
):
    """Show recent search history (if available)"""
    
    # 将来実装用のプレースホルダー
    console.print("[dim]Search history is not yet implemented[/dim]")
