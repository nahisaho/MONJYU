# MONJYU CLI - Citation Commands
"""
FEAT-008: CLI - citation コマンド群
引用ネットワーク可視化・分析
"""

from pathlib import Path
from typing import Optional
import json

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel

from monjyu.cli.main import OutputFormat, get_monjyu, print_error

console = Console()
citation_app = typer.Typer(help="Citation network commands")


@citation_app.command("chain")
def citation_chain(
    document_id: str = typer.Argument(..., help="Document ID"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
    depth: int = typer.Option(
        2, "--depth", "-d", help="Citation chain depth"
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", "-o", help="Output format"
    ),
):
    """Show citation chain for a document"""
    
    try:
        monjyu = get_monjyu(config)
        doc = monjyu.get_document(document_id)
        
        if doc is None:
            print_error(f"Document not found: {document_id}")
            raise typer.Exit(1)
        
        # 引用ネットワーク取得
        citation_manager = monjyu.get_citation_network()
        
        if citation_manager is None:
            console.print("[yellow]Citation network not available[/yellow]")
            raise typer.Exit(0)
        
        # 引用チェーン構築
        cites = []
        cited_by = []
        
        try:
            # このドキュメントが引用している文献
            cites = citation_manager.get_references(document_id)[:10]
            # このドキュメントを引用している文献
            cited_by = citation_manager.get_citations(document_id)[:10]
        except Exception:
            pass
        
        if output == OutputFormat.json:
            console.print_json(json.dumps({
                "document_id": document_id,
                "title": doc.title,
                "cites": [c.target_id for c in cites],
                "cited_by": [c.source_id for c in cited_by],
            }))
        else:
            # ツリー表示
            tree = Tree(f"[bold]{doc.title}[/bold]")
            
            # 引用している文献
            if cites:
                cites_branch = tree.add("[cyan]📚 References (cites)[/cyan]")
                for ref in cites[:10]:
                    ref_doc = monjyu.get_document(ref.target_id)
                    title = ref_doc.title if ref_doc else ref.target_id
                    cites_branch.add(f"[dim]{title}[/dim]")
            else:
                tree.add("[dim]📚 References: None found[/dim]")
            
            # 引用されている文献
            if cited_by:
                cited_branch = tree.add("[green]📖 Cited by[/green]")
                for cite in cited_by[:10]:
                    cite_doc = monjyu.get_document(cite.source_id)
                    title = cite_doc.title if cite_doc else cite.source_id
                    cited_branch.add(f"[dim]{title}[/dim]")
            else:
                tree.add("[dim]📖 Cited by: None found[/dim]")
            
            console.print(tree)
            console.print(
                f"\n[dim]References: {len(cites)} | Cited by: {len(cited_by)}[/dim]"
            )
    
    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Failed to get citation chain: {e}")
        raise typer.Exit(1)


@citation_app.command("related")
def citation_related(
    document_id: str = typer.Argument(..., help="Document ID"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
    top_k: int = typer.Option(
        10, "--top-k", "-k", help="Number of related papers"
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", "-o", help="Output format"
    ),
):
    """Find related papers based on citation network"""
    
    try:
        monjyu = get_monjyu(config)
        doc = monjyu.get_document(document_id)
        
        if doc is None:
            print_error(f"Document not found: {document_id}")
            raise typer.Exit(1)
        
        # 関連論文を取得（引用ネットワークベース）
        citation_manager = monjyu.get_citation_network()
        
        if citation_manager is None:
            console.print("[yellow]Citation network not available[/yellow]")
            raise typer.Exit(0)
        
        # 共引用・書誌結合による関連論文
        try:
            related = citation_manager.find_co_citation_papers(document_id, top_k)
        except Exception:
            related = []
        
        if output == OutputFormat.json:
            console.print_json(json.dumps([
                {
                    "document_id": r[0],
                    "score": r[1],
                }
                for r in related
            ]))
        else:
            if not related:
                console.print(
                    f"[yellow]No related papers found for document: {document_id}[/yellow]"
                )
                return
            
            table = Table(title=f"Related Papers to: {doc.title[:40]}...")
            table.add_column("Title", max_width=50)
            table.add_column("Relationship", justify="center")
            table.add_column("Score", justify="right")
            
            for doc_id, score in related:
                related_doc = monjyu.get_document(doc_id)
                title = related_doc.title if related_doc else doc_id
                
                table.add_row(
                    title[:50] + ("..." if len(title) > 50 else ""),
                    "co-citation",
                    f"{score:.3f}",
                )
            
            console.print(table)
    
    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Failed to find related papers: {e}")
        raise typer.Exit(1)


@citation_app.command("top")
def citation_top(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
    top_k: int = typer.Option(
        10, "--top-k", "-k", help="Number of top papers"
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", "-o", help="Output format"
    ),
):
    """Show most influential papers in the corpus"""
    
    try:
        monjyu = get_monjyu(config)
        citation_manager = monjyu.get_citation_network()
        
        if citation_manager is None:
            console.print("[yellow]Citation network not available[/yellow]")
            raise typer.Exit(0)
        
        # 影響力の高い論文を取得
        try:
            top_papers = citation_manager.get_most_influential(top_k)
        except Exception:
            top_papers = []
        
        if output == OutputFormat.json:
            console.print_json(json.dumps([
                {
                    "document_id": m.document_id,
                    "citation_count": m.citation_count,
                    "pagerank": m.pagerank,
                    "influence_score": m.influence_score,
                }
                for m in top_papers
            ]))
        else:
            if not top_papers:
                console.print("[yellow]No papers with citation metrics found[/yellow]")
                return
            
            table = Table(title="Most Influential Papers")
            table.add_column("Rank", justify="right", width=5)
            table.add_column("Title", max_width=40)
            table.add_column("Citations", justify="right", width=10)
            table.add_column("PageRank", justify="right", width=10)
            table.add_column("Influence", justify="right", width=10)
            
            for i, metrics in enumerate(top_papers, 1):
                doc = monjyu.get_document(metrics.document_id)
                title = doc.title if doc else metrics.document_id
                
                table.add_row(
                    str(i),
                    title[:40] + ("..." if len(title) > 40 else ""),
                    str(metrics.citation_count),
                    f"{metrics.pagerank:.4f}",
                    f"{metrics.influence_score:.4f}",
                )
            
            console.print(table)
    
    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Failed to get top papers: {e}")
        raise typer.Exit(1)


@citation_app.command("stats")
def citation_stats(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", "-o", help="Output format"
    ),
):
    """Show citation network statistics"""
    
    try:
        monjyu = get_monjyu(config)
        citation_manager = monjyu.get_citation_network()
        
        if citation_manager is None:
            console.print("[yellow]Citation network not available[/yellow]")
            raise typer.Exit(0)
        
        # 統計情報取得
        try:
            stats = citation_manager.get_network_stats()
        except Exception:
            stats = {}
        
        if output == OutputFormat.json:
            console.print_json(json.dumps(stats))
        else:
            console.print(Panel.fit(
                f"[bold]Citation Network Statistics[/bold]\n\n"
                f"Documents:        {stats.get('document_count', 0):,}\n"
                f"Citation Edges:   {stats.get('edge_count', 0):,}\n"
                f"Avg. Citations:   {stats.get('avg_citations', 0):.2f}\n"
                f"Avg. References:  {stats.get('avg_references', 0):.2f}\n"
                f"Network Density:  {stats.get('density', 0):.4f}",
                title="Network Stats",
                border_style="cyan",
            ))
    
    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Failed to get citation stats: {e}")
        raise typer.Exit(1)
