# MONJYU CLI - Main Application
"""
FEAT-008: CLI (Command Line Interface)
メインアプリケーション構造
"""

from pathlib import Path
from typing import Optional
from enum import Enum

import typer
from rich.console import Console
from rich.panel import Panel

# === アプリケーション初期化 ===

app = typer.Typer(
    name="monjyu",
    help="MONJYU - Academic Paper RAG System using Progressive GraphRAG",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()


# === 出力フォーマット ===

class OutputFormat(str, Enum):
    """出力フォーマット"""
    text = "text"
    json = "json"


# === ユーティリティ関数 ===

def get_monjyu(config_path: Optional[Path] = None):
    """MONJYUインスタンスを取得
    
    Args:
        config_path: 設定ファイルパス（Noneの場合はデフォルト）
    
    Returns:
        MONJYU: 初期化済みインスタンス
    """
    from monjyu.api import MONJYU
    
    # 設定ファイルを探索
    search_paths = [
        config_path,
        Path("./monjyu.yaml"),
        Path("./monjyu.yml"),
        Path("./config/monjyu.yaml"),
    ]
    
    for path in search_paths:
        if path and path.exists():
            return MONJYU(path)
    
    # デフォルト設定
    return MONJYU()


def print_error(message: str):
    """エラーメッセージを表示"""
    console.print(f"[red]✗ Error:[/red] {message}")


def print_success(message: str):
    """成功メッセージを表示"""
    console.print(f"[green]✓[/green] {message}")


def print_warning(message: str):
    """警告メッセージを表示"""
    console.print(f"[yellow]⚠[/yellow] {message}")


# === バージョンコマンド ===

@app.command()
def version():
    """Show version information"""
    try:
        from monjyu import __version__
        ver = __version__
    except ImportError:
        ver = "0.1.0"
    
    console.print(Panel.fit(
        f"[bold cyan]MONJYU[/bold cyan] v{ver}\n"
        "[dim]Academic Paper RAG System using Progressive GraphRAG[/dim]",
        border_style="cyan"
    ))


# === initコマンド ===

@app.command()
def init(
    path: Path = typer.Argument(
        Path("."),
        help="Project directory to initialize",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing files"
    ),
):
    """Initialize a new MONJYU project
    
    Creates the following structure:
    - monjyu.yaml (configuration file)
    - output/ (index output directory)
    - papers/ (documents directory)
    """
    from monjyu.cli.commands.config_cmd import DEFAULT_CONFIG
    
    project_path = path.resolve()
    config_file = project_path / "monjyu.yaml"
    output_dir = project_path / "output"
    papers_dir = project_path / "papers"
    
    # 既存チェック
    if config_file.exists() and not force:
        print_warning(f"Project already initialized: {config_file}")
        console.print("Use --force to reinitialize")
        raise typer.Exit(1)
    
    try:
        # ディレクトリ作成
        project_path.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(exist_ok=True)
        papers_dir.mkdir(exist_ok=True)
        
        # 設定ファイル作成
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CONFIG)
        
        # .gitignore 作成
        gitignore_file = project_path / ".gitignore"
        if not gitignore_file.exists():
            with open(gitignore_file, "w", encoding="utf-8") as f:
                f.write("# MONJYU\noutput/\n*.pyc\n__pycache__/\n.env\n")
        
        console.print(Panel.fit(
            f"[green]✓ MONJYU project initialized![/green]\n\n"
            f"[bold]Created:[/bold]\n"
            f"  📄 {config_file.relative_to(project_path.parent)}\n"
            f"  📁 {output_dir.relative_to(project_path.parent)}/\n"
            f"  📁 {papers_dir.relative_to(project_path.parent)}/\n\n"
            f"[bold]Next steps:[/bold]\n"
            f"  1. Add PDF papers to [cyan]papers/[/cyan]\n"
            f"  2. Edit [cyan]monjyu.yaml[/cyan] if needed\n"
            f"  3. Build index: [cyan]monjyu index build papers/[/cyan]\n"
            f"  4. Search: [cyan]monjyu query \"your question\"[/cyan]",
            title="Project Initialized",
            border_style="green"
        ))
        
    except Exception as e:
        print_error(f"Failed to initialize project: {e}")
        raise typer.Exit(1)


# === queryコマンド（searchのエイリアス） ===

@app.command()
def query(
    question: str = typer.Argument(..., help="Search query"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
    mode: str = typer.Option(
        "lazy", "--mode", "-m",
        help="Search mode: vector, lazy, local, global, auto"
    ),
    top_k: int = typer.Option(
        10, "--top-k", "-k", help="Number of results"
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", "-o", help="Output format"
    ),
):
    """Execute a search query (shortcut for 'search')
    
    Examples:
        monjyu query "What is transformer architecture?"
        monjyu query "深層学習の最新手法" --mode local
    """
    import json
    from rich.markdown import Markdown
    from monjyu.api import SearchMode
    
    try:
        monjyu = get_monjyu(config)
        
        # 検索モード変換
        mode_map = {
            "vector": SearchMode.VECTOR,
            "lazy": SearchMode.LAZY,
            "local": SearchMode.LOCAL,
            "global": SearchMode.GLOBAL,
            "auto": SearchMode.AUTO,
        }
        
        search_mode = mode_map.get(mode.lower())
        if search_mode is None:
            print_error(f"Invalid search mode: {mode}")
            console.print("Valid modes: vector, lazy, local, global, auto")
            raise typer.Exit(1)
        
        # 検索実行
        with console.status("[bold green]Searching...", spinner="dots"):
            result = monjyu.search(question, mode=search_mode, top_k=top_k)
        
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
                console.print("\n[bold]Sources:[/bold]")
                for i, citation in enumerate(result.citations[:5], 1):
                    title = citation.title or citation.doc_id
                    console.print(f"  [{i}] [cyan]{title}[/cyan]")
            
            # メタデータ
            console.print(
                f"\n[dim]Time: {result.total_time_ms:.0f}ms[/dim]"
            )
    
    except Exception as e:
        print_error(f"Search failed: {e}")
        raise typer.Exit(1)


# === サブコマンドのインポートとアタッチ ===

def attach_commands():
    """サブコマンドをアタッチ"""
    from monjyu.cli.commands import (
        index_app,
        search_app,
        document_app,
        citation_app,
        config_app,
    )
    
    app.add_typer(index_app, name="index")
    app.add_typer(search_app, name="search")
    app.add_typer(document_app, name="document")
    app.add_typer(citation_app, name="citation")
    app.add_typer(config_app, name="config")


# コマンドをアタッチ
try:
    attach_commands()
except ImportError:
    # コマンドモジュールがまだない場合はスキップ
    pass
