import { useEffect, useRef, useState } from "react";
import cytoscape from "cytoscape";
import InfoTooltip from "./InfoTooltip";

/**
 * StructuralLens - GNN Explainer Subgraph Visualization
 * 
 * Design Principle: Clarity-First, No Visual Bias
 * - All nodes are circles with fixed size
 * - All edges have fixed width
 * - Semantic meaning conveyed by COLOR ONLY
 * - Importance shown in tooltips, not visuals
 * - No overlap between nodes/edges
 */
function StructuralLens({ gnnFeatures, topEdges, graphElements, graphStats, transactionId, senderAccount, receiverAccount }) {
    const containerRef = useRef(null);
    const cyRef = useRef(null);
    const [hoveredElement, setHoveredElement] = useState(null);
    const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

    useEffect(() => {
        if (!containerRef.current) return;

        // Use backend graphElements - complete unfiltered subgraph
        let elements;
        if (graphElements && graphElements.nodes && graphElements.edges) {
            elements = {
                nodes: graphElements.nodes,
                edges: graphElements.edges
            };
        } else {
            // Minimal fallback
            console.warn("Backend graphElements missing - using minimal fallback");
            const anchorId = `tx_${transactionId}`;
            elements = {
                nodes: [
                    { data: { id: anchorId, label: anchorId, type: "transaction", is_anchor: true } }
                ],
                edges: []
            };
            if (senderAccount) {
                elements.nodes.push({ data: { id: senderAccount, label: senderAccount, type: "account" } });
                elements.edges.push({ data: { id: `e_s`, source: senderAccount, target: anchorId, relation: "sends" } });
            }
            if (receiverAccount) {
                elements.nodes.push({ data: { id: receiverAccount, label: receiverAccount, type: "account" } });
                elements.edges.push({ data: { id: `e_r`, source: anchorId, target: receiverAccount, relation: "receives" } });
            }
        }

        // Initialize Cytoscape with Clarity-First Styling
        const cy = cytoscape({
            container: containerRef.current,
            elements: elements,
            style: [
                // === ANCHOR TRANSACTION (Deep Red) ===
                {
                    selector: "node[is_anchor]",
                    style: {
                        "shape": "ellipse",  // MUST be circle
                        "width": 90,
                        "height": 90,
                        "background-color": "#E53935",  // Deep red
                        "border-width": 3,
                        "border-color": "#B71C1C",
                        "label": "data(label)",
                        "color": "#FFFFFF",
                        "text-valign": "center",
                        "text-halign": "center",
                        "font-size": "12px",
                        "font-weight": "600",
                        "text-wrap": "wrap",
                        "text-max-width": "80px",
                        "z-index": 10
                    }
                },
                // === CONNECTED TRANSACTIONS (Orange) ===
                {
                    selector: "node[type='transaction'][!is_anchor]",
                    style: {
                        "shape": "ellipse",
                        "width": 80,
                        "height": 80,
                        "background-color": "#FB8C00",  // Orange
                        "border-width": 2,
                        "border-color": "#EF6C00",
                        "label": "data(label)",
                        "color": "#FFFFFF",
                        "text-valign": "center",
                        "text-halign": "center",
                        "font-size": "11px",
                        "font-weight": "500",
                        "text-wrap": "wrap",
                        "text-max-width": "70px"
                    }
                },
                // === ACCOUNT NODES (Blue) ===
                {
                    selector: "node[type='account']",
                    style: {
                        "shape": "ellipse",
                        "width": 80,
                        "height": 80,
                        "background-color": "#1E88E5",  // Blue
                        "border-width": 2,
                        "border-color": "#0D47A1",
                        "label": "data(label)",
                        "color": "#FFFFFF",
                        "text-valign": "center",
                        "text-halign": "center",
                        "font-size": "10px",
                        "font-weight": "500",
                        "text-wrap": "wrap",
                        "text-max-width": "70px"
                    }
                },
                // === SENDER EDGES (account → transaction) ===
                {
                    selector: "edge[relation='sends']",
                    style: {
                        "width": 3,  // Increased for better visibility
                        "line-color": "#424242",  // Dark gray
                        "target-arrow-color": "#424242",
                        "target-arrow-shape": "triangle",
                        "curve-style": "bezier",
                        "arrow-scale": 1.2,
                        "opacity": 1  // No transparency
                    }
                },
                // === RECEIVER EDGES (transaction → account) ===
                {
                    selector: "edge[relation='receives']",
                    style: {
                        "width": 3,  // Increased for better visibility
                        "line-color": "#616161",  // Lighter gray
                        "target-arrow-color": "#616161",
                        "target-arrow-shape": "triangle",
                        "curve-style": "bezier",
                        "arrow-scale": 1.2,
                        "opacity": 1  // No transparency
                    }
                },
                // === FALLBACK FOR EDGES WITHOUT RELATION ===
                {
                    selector: "edge[!relation]",
                    style: {
                        "width": 3,
                        "line-color": "#757575",
                        "target-arrow-color": "#757575",
                        "target-arrow-shape": "triangle",
                        "curve-style": "bezier",
                        "arrow-scale": 1.2,
                        "opacity": 1
                    }
                },
                // === HOVER STATE ===
                {
                    selector: "node:active, node:selected",
                    style: {
                        "border-width": 4,
                        "border-color": "#FFD700"  // Gold highlight
                    }
                }
            ],
            layout: {
                name: "cose",
                animate: false,  // No jitter
                randomize: false,  // Deterministic
                nodeRepulsion: 8000,  // Prevent overlap
                idealEdgeLength: 120,  // Spacing
                nodeOverlap: 20,  // Anti-overlap buffer
                padding: 50,
                gravity: 0.5,
                numIter: 1000,  // Converge fully
                initialTemp: 200,
                coolingFactor: 0.95,
                minTemp: 1.0
            },
            // Prevent user-caused layout jitter
            userZoomingEnabled: true,
            userPanningEnabled: true,
            boxSelectionEnabled: false
        });

        // === NODE HOVER ===
        cy.on("mouseover", "node", (evt) => {
            const node = evt.target;
            const data = node.data();
            const position = node.renderedPosition();

            let role = "Node";
            if (data.is_anchor) {
                role = "Anchor Transaction";
            } else if (data.type === "transaction") {
                role = "Connected Transaction";
            } else if (data.type === "account") {
                role = "Account";
            }

            setHoveredElement({
                type: "node",
                id: data.label || data.id,
                nodeType: data.type,
                role: role,
                is_anchor: data.is_anchor || false,
                importance: data.importance  // Raw value
            });

            setTooltipPos({ x: position.x, y: position.y + 50 });

            // Highlight on hover
            node.style("border-width", 5);
            node.style("border-color", "#FFD700");
        });

        cy.on("mouseout", "node", (evt) => {
            const node = evt.target;
            const data = node.data();

            // Restore original border
            if (data.is_anchor) {
                node.style("border-width", 3);
                node.style("border-color", "#B71C1C");
            } else {
                node.style("border-width", 2);
                if (data.type === "transaction") {
                    node.style("border-color", "#EF6C00");
                } else {
                    node.style("border-color", "#0D47A1");
                }
            }

            setHoveredElement(null);
            setTooltipPos({ x: 0, y: 0 });
        });

        // === EDGE HOVER ===
        cy.on("mouseover", "edge", (evt) => {
            const edge = evt.target;
            const data = edge.data();
            const midpoint = edge.renderedMidpoint();

            setHoveredElement({
                type: "edge",
                relation: data.relation || "connected",
                source: data.source,
                target: data.target,
                weight: data.weight,  // Raw value
                role: data.role
            });

            setTooltipPos({ x: midpoint.x, y: midpoint.y + 20 });

            // Highlight on hover
            edge.style("width", 5);
            edge.style("line-color", "#FFD700");
            edge.style("target-arrow-color", "#FFD700");
        });

        cy.on("mouseout", "edge", (evt) => {
            const edge = evt.target;
            const data = edge.data();

            // Restore original style
            edge.style("width", 3);
            if (data.relation === "sends") {
                edge.style("line-color", "#424242");
                edge.style("target-arrow-color", "#424242");
            } else if (data.relation === "receives") {
                edge.style("line-color", "#616161");
                edge.style("target-arrow-color", "#616161");
            } else {
                edge.style("line-color", "#757575");
                edge.style("target-arrow-color", "#757575");
            }

            setHoveredElement(null);
            setTooltipPos({ x: 0, y: 0 });
        });

        cyRef.current = cy;
        return () => cy.destroy();
    }, [graphElements, transactionId, senderAccount, receiverAccount]);

    const handleFit = () => {
        if (cyRef.current) {
            cyRef.current.fit(50);  // 50px padding
            cyRef.current.center();
        }
    };


    return (
        <div className="graph-panel" style={{ marginBottom: 0 }}>
            <div className="panel-header">
                <span className="panel-title">STRUCTURAL LENS (GNN)</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <button
                        onClick={handleFit}
                        style={{
                            padding: '0.25rem 0.5rem',
                            fontSize: '0.7rem',
                            background: 'var(--bg-dark)',
                            border: '1px solid var(--border-color)',
                            borderRadius: '4px',
                            color: 'var(--text-secondary)',
                            cursor: 'pointer'
                        }}
                        title="Fit graph to view"
                    >
                        ⟳ Fit
                    </button>
                    <InfoTooltip tooltipKey="structuralLens" />
                </div>
            </div>

            {/* === COLOR LEGEND (Semantic Only) === */}
            <div className="graph-legend" style={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: '1rem',
                padding: '0.5rem 0.75rem',
                fontSize: '0.75rem',
                color: 'var(--text-secondary)',
                borderBottom: '1px solid var(--border-color)',
                background: 'rgba(0,0,0,0.2)'
            }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    <span style={{ width: '14px', height: '14px', borderRadius: '50%', background: '#E53935', border: '2px solid #B71C1C' }}></span>
                    Anchor TX
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    <span style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#FB8C00', border: '1.5px solid #EF6C00' }}></span>
                    Transaction
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    <span style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#1E88E5', border: '1.5px solid #0D47A1' }}></span>
                    Account
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', marginLeft: 'auto' }}>
                    <span style={{ fontSize: '0.65rem', fontStyle: 'italic', color: 'var(--text-muted)' }}>
                        Size & width are fixed • Importance shown in tooltips
                    </span>
                </span>
            </div>

            <div style={{ position: "relative" }}>
                <div ref={containerRef} className="cytoscape-container" />

                {/* === TOOLTIP (Raw Importance Values) === */}
                {hoveredElement && (
                    <div
                        className="graph-hover-tooltip"
                        style={{
                            position: 'absolute',
                            left: `${tooltipPos.x}px`,
                            top: `${tooltipPos.y}px`,
                            transform: 'translateX(-50%)',
                            pointerEvents: 'none',
                            zIndex: 1000,
                            minWidth: '200px'
                        }}
                    >
                        {hoveredElement.type === "node" ? (
                            <>
                                <div className="tooltip-row">
                                    <span className="tooltip-label">Type:</span>
                                    <span className="tooltip-value">{hoveredElement.nodeType}</span>
                                </div>
                                <div className="tooltip-row">
                                    <span className="tooltip-label">Role:</span>
                                    <span className="tooltip-value">{hoveredElement.role}</span>
                                </div>
                                <div className="tooltip-row">
                                    <span className="tooltip-label">ID:</span>
                                    <span className="tooltip-value" style={{ fontSize: '0.85em', wordBreak: 'break-all' }}>
                                        {hoveredElement.id}
                                    </span>
                                </div>
                                {hoveredElement.is_anchor && (
                                    <div className="tooltip-row">
                                        <span className="tooltip-value" style={{ color: '#FFD700', fontWeight: 'bold' }}>
                                            ⭐ Anchor (Under Explanation)
                                        </span>
                                    </div>
                                )}
                                {hoveredElement.importance != null && (
                                    <div className="tooltip-row" style={{ borderTop: '1px solid var(--border-color)', paddingTop: '0.25rem', marginTop: '0.25rem' }}>
                                        <span className="tooltip-label">Importance:</span>
                                        <span className="tooltip-value">{hoveredElement.importance.toFixed(6)}</span>
                                    </div>
                                )}
                            </>
                        ) : (
                            <>
                                <div className="tooltip-row">
                                    <span className="tooltip-label">Relation:</span>
                                    <span className="tooltip-value">{hoveredElement.relation}</span>
                                </div>
                                <div className="tooltip-row">
                                    <span className="tooltip-label">Direction:</span>
                                    <span className="tooltip-value">
                                        {hoveredElement.source} → {hoveredElement.target}
                                    </span>
                                </div>
                                {hoveredElement.weight != null && (
                                    <>
                                        <div className="tooltip-row" style={{ borderTop: '1px solid var(--border-color)', paddingTop: '0.25rem', marginTop: '0.25rem' }}>
                                            <span className="tooltip-label">Weight:</span>
                                            <span className="tooltip-value">{hoveredElement.weight.toFixed(6)}</span>
                                        </div>
                                        <div className="tooltip-row">
                                            <span className="tooltip-value" style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                                                GNN explainer edge importance
                                            </span>
                                        </div>
                                    </>
                                )}
                            </>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

export default StructuralLens;

