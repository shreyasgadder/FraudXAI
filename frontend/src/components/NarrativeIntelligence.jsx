import { useState } from "react";
import { api } from "../utils/api";
import InfoTooltip from "./InfoTooltip";

function NarrativeIntelligence({
    transactionId,
    metadata,
    explanations,
    narrative,
    systemExplanation,
    pattern,
    trust
}) {
    const [aiNarrative, setAiNarrative] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const getSystemNarrative = () => {
        if (systemExplanation) return systemExplanation;
        if (narrative) return narrative;
        if (!metadata) return "Generating explanation...";

        const parts = [];
        const prob = (metadata.fraud_probability || 0) * 100;
        const riskLevel = metadata.risk_level || "unknown";
        parts.push(`This transaction is flagged as ${riskLevel.toLowerCase()} risk with ${prob.toFixed(1)}% fraud probability.`);

        if (explanations?.features?.length > 0) {
            const top = explanations.features[0];
            const direction = top.Importance > 0 ? "increases" : "decreases";
            parts.push(`Primary driver: ${top.Feature} (${direction} risk by ${Math.abs(top.Importance).toFixed(3)}).`);
        }
        return parts.join(" ");
    };

    const handleGenerateNarrative = async () => {
        setLoading(true);
        setError(null);
        try {
            const enhancedMetadata = { ...metadata, graph: metadata.graph || {} };
            const result = await api.generateNarrative(
                transactionId,
                enhancedMetadata,
                explanations,
                pattern,
                trust,
                systemExplanation
            );
            setAiNarrative(result.narrative);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const systemNarrative = getSystemNarrative();

    // --- ROBUST FORMATTING LOGIC ---
    const renderFormattedText = (text) => {
        if (!text) return null;

        // 1. Pre-process: Ensure headers have newlines before them.
        // This fixes the "one big paragraph" issue by forcing breaks before the emojis.
        // We look for the pattern **[Emoji] and add a newline before it.
        let processedText = text.replace(/(\*\*🚨|\*\*🔍|\*\*🛡️)/g, '\n$1');

        // 2. Split by newlines to process each line individually
        return processedText.split(/\r?\n/).map((line, lineIndex) => {
            if (!line.trim()) return <div key={lineIndex} style={{ height: '0.5rem' }} />;

            // 3. Regex Split for Bolding
            // The regex /\*\*([^*]+)\*\*/g captures the text INSIDE the stars.
            // "A **B** C".split(...) -> ["A ", "B", " C"]
            // Even indices (0, 2) are normal text. Odd indices (1) are the captured bold text.
            const parts = line.split(/\*\*([^*]+)\*\*/g);

            const lineContent = parts.map((part, partIndex) => {
                if (partIndex % 2 === 1) {
                    // This is the captured text from inside the **...**
                    return (
                        <strong key={partIndex} style={{ color: "var(--text-primary)" }}>
                            {part}
                        </strong>
                    );
                }
                // Normal text
                return part;
            });

            // 4. List Item Styling
            const isListItem = line.trim().startsWith('-');
            const style = {
                margin: '0 0 0.25rem 0',
                paddingLeft: isListItem ? '1rem' : '0',
                color: "var(--text-secondary)",
                lineHeight: "1.5"
            };

            return <div key={lineIndex} style={style}>{lineContent}</div>;
        });
    };

    return (
        <div className="narrative-intelligence">
            <div className="narrative-panel">
                <h4>
                    SYSTEM EXPLANATION (Rule-Based)
                    <InfoTooltip tooltipKey="systemExplanation" />
                </h4>
                <div className="narrative-content">
                    {renderFormattedText(systemNarrative)}
                </div>
            </div>

            <div className="narrative-panel">
                <h4>
                    AI-GENERATED NARRATIVE (Gemini 2.5 Flash)
                    <InfoTooltip tooltipKey="aiNarrative" />
                </h4>

                {!aiNarrative && !loading && !error && (
                    <button className="btn-generate" onClick={handleGenerateNarrative}>
                        ✨ Generate AI Narrative
                    </button>
                )}

                {loading && <p style={{ color: "var(--text-muted)", fontStyle: 'italic' }}>Analyzing graph patterns and risk factors...</p>}

                {error && <p style={{ color: "var(--risk-high)" }}>❌ {error}</p>}

                {aiNarrative && (
                    <>
                        <div className="ai-narrative-box" style={{
                            background: "rgba(0,0,0,0.02)",
                            padding: "1rem",
                            borderRadius: "8px",
                            border: "1px solid var(--border-color)"
                        }}>
                            {renderFormattedText(aiNarrative)}
                        </div>
                        <button
                            className="btn-link"
                            onClick={() => setAiNarrative(null)}
                            style={{ marginTop: "0.5rem", fontSize: "0.85rem" }}
                        >
                            Regenerate
                        </button>
                    </>
                )}
            </div>
        </div>
    );
}

export default NarrativeIntelligence;