import { useState, useEffect } from "react";
import { api } from "../utils/api";
import TrustBadge from "./TrustBadge";
import InfoTooltip from "./InfoTooltip";
import StructuralLens from "./StructuralLens";
import DriversList from "./DriversList";
import NarrativeIntelligence from "./NarrativeIntelligence";
import FeaturesImportance from "./FeaturesImportance";

function InvestigationModal({ transaction, onClose }) {
    const [explanation, setExplanation] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const abortController = new AbortController();

        const loadExplanation = async () => {
            setLoading(true);
            setError(null);

            try {
                const data = await api.explainTransaction(
                    transaction.transaction_id,
                    abortController.signal
                );
                console.log("Explanation data:", data);
                setExplanation(data);
            } catch (err) {
                // Ignore abort errors
                if (err.name === 'AbortError') {
                    console.log('Request cancelled');
                    return;
                }
                setError(err.message);
            } finally {
                if (!abortController.signal.aborted) {
                    setLoading(false);
                }
            }
        };

        loadExplanation();

        // Cleanup: abort the request if component unmounts or transaction changes
        return () => abortController.abort();
    }, [transaction.transaction_id]);

    const getRiskClass = (level) => {
        switch (level?.toLowerCase()) {
            case "high": return "high";
            case "medium": return "medium";
            default: return "low";
        }
    };

    const getRiskIcon = (level) => {
        switch (level?.toLowerCase()) {
            case "high": return "🔴";
            case "medium": return "🟡";
            default: return "🟢";
        }
    };

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                <div className="modal-header">
                    <h2>🔍 Investigation: Transaction #{transaction.transaction_id}</h2>
                    <button className="modal-close" onClick={onClose}>✕</button>
                </div>

                {loading && <div className="loading">Generating explanation...</div>}

                {error && (
                    <div className="error-message">
                        ❌ Failed to generate explanation: {error}
                    </div>
                )}

                {explanation && (
                    <div className="modal-body">
                        {/* Warnings */}
                        {explanation.warnings?.length > 0 && (
                            <div className="warnings-section">
                                {explanation.warnings.map((w, i) => (
                                    <div key={i} className="warning-message">⚠️ {w}</div>
                                ))}
                            </div>
                        )}


                        {/* Pattern details are shown in the pattern-banner below */}

                        {/* Risk Header Section */}
                        <div className="risk-header-section">
                            <div className="badges-row">
                                {/* Risk Badge with Info inside */}
                                <div className={`score-badge risk-badge ${getRiskClass(explanation.metadata?.risk_level)}`}>
                                    <span className="badge-icon">{getRiskIcon(explanation.metadata?.risk_level)}</span>
                                    <div className="badge-content">
                                        <span className="badge-label">
                                            {explanation.metadata?.risk_level || "UNKNOWN"} RISK
                                        </span>
                                        <span className="badge-value">
                                            {((explanation.metadata?.fraud_probability || 0) * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                    <InfoTooltip tooltipKey="riskScore" />
                                </div>

                                {/* Trust Badge - Use enhanced trust info if available */}
                                <TrustBadge
                                    score={explanation.bac_score || 0}
                                    trustInfo={explanation.trust}
                                />
                            </div>

                            <div className="metadata-row">
                                <div className="metadata-item">
                                    <span className="label">Amount:</span>
                                    <span className="value">
                                        ${(transaction.amount || 0).toLocaleString()}
                                    </span>
                                </div>
                                <div className="metadata-item">
                                    <span className="label">Type:</span>
                                    <span className="value">{transaction.type || "N/A"}</span>
                                </div>
                                <div className="metadata-item">
                                    <span className="label">Hour:</span>
                                    <span className="value">
                                        {transaction.hour_of_day != null
                                            ? `${transaction.hour_of_day.toString().padStart(2, '0')}:00`
                                            : "N/A"}
                                    </span>
                                </div>
                                <div className="metadata-item">
                                    <span className="label">Sender:</span>
                                    <span className="value">{transaction.sender_id || "N/A"}</span>
                                </div>
                                <div className="metadata-item">
                                    <span className="label">Receiver:</span>
                                    <span className="value">{transaction.receiver_id || "N/A"}</span>
                                </div>
                            </div>
                        </div>

                        {/* Network Analysis Section */}
                        <div className="evidence-section">
                            {/* GNN Graph - Use new graph elements if available */}
                            <StructuralLens
                                gnnFeatures={explanation.explanations?.gnn_features}
                                topEdges={explanation.explanations?.top_edges}
                                graphElements={explanation.graph?.elements}
                                graphStats={explanation.graph?.stats}
                                transactionId={transaction.transaction_id}
                                senderAccount={transaction.sender_id}
                                receiverAccount={transaction.receiver_id}
                            />

                            {/* Graph Stats */}
                            {explanation.graph?.stats && (
                                <div className="graph-stats-bar">
                                    <span>{explanation.graph.stats.num_accounts || 0} accounts</span>
                                    <span>•</span>
                                    <span>{explanation.graph.stats.num_transactions || 0} transactions</span>
                                    <span>•</span>
                                    <span>{explanation.graph.stats.num_edges || 0} edges</span>
                                </div>
                            )}
                            {/* Unified Features Importance - Above drivers grid */}
                            {explanation.features && (
                                <FeaturesImportance features={explanation.features} />
                            )}

                        </div>

                        {/* Narrative Section */}
                        <div className="narrative-section">
                            <NarrativeIntelligence
                                transactionId={transaction.transaction_id}
                                metadata={{
                                    ...explanation.metadata,
                                    graph: explanation.graph  // Add full graph object
                                }}
                                explanations={{
                                    ...explanation.explanations,
                                    features: explanation.features  // Add unified features
                                }}
                                narrative={explanation.narrative}
                                systemExplanation={explanation.system_explanation}
                                pattern={explanation.pattern}
                                trust={explanation.trust}
                            />
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default InvestigationModal;
