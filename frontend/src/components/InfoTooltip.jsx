import { useState } from "react";

const TOOLTIP_TEXT = {
    riskScore: `The model's estimated probability that this transaction is fraudulent.`,

    featuresImportance: `This list ranks features by their influence on the fraud verdict, combining individual transaction data with broader network patterns. High-ranking features represent the primary drivers of the risk score, while lower-ranking features provided secondary context.`,

    trustScore: `Trust Score reflects how reliable this explanation is.

It combines:
• Faithfulness to model behavior
• Sparsity (clarity of drivers)
• Stability under perturbation

Higher trust means the explanation is more consistent and reliable.`,

    structuralLens: `This graph shows how the transaction is connected to other accounts and transactions.

• Nodes represent accounts and transactions
• Edges represent money flow relationships
• The highlighted transaction is the one under investigation

Hover on nodes or edges to see their contribution.`,

    tabularDrivers: `These features explain the decision using transaction data only.

• Positive values increase fraud risk
• Negative values decrease fraud risk
• Features are shown in backend-provided order`,

    structuralDrivers: `These features explain the decision using network behavior.

They capture patterns such as:
• Repeated interactions
• High fan-in or fan-out activity
• Abnormal transaction timing`,

    systemExplanation: `This explanation is generated deterministically using model outputs.
It does not use AI and is fully auditable.`,

    aiNarrative: `This explanation is generated using Gemini 2.5 Flash to help non-technical users understand the decision.

It does not influence the risk score.`
};

function InfoTooltip({ tooltipKey, trustInfo }) {
    const [show, setShow] = useState(false);
    const content = TOOLTIP_TEXT[tooltipKey] || "";

    // Enhanced tooltip for trust score showing component breakdown
    const getTrustContent = () => {
        if (tooltipKey !== "trustScore" || !trustInfo) return content;

        const components = trustInfo.components || {};
        const faithfulness = ((components.faithfulness || 0) * 100).toFixed(0);
        const sparsity = ((components.sparsity || 0) * 100).toFixed(0);
        const stability = ((components.stability || 0) * 100).toFixed(0);

        return `Trust Score reflects how reliable this explanation is.

It combines:
• Faithfulness to model behavior: ${faithfulness}%
• Sparsity (clarity of drivers): ${sparsity}%
• Stability under perturbation: ${stability}%

Higher trust means the explanation is more consistent and reliable.`;
    };

    return (
        <span
            className="info-tooltip-inline"
            onMouseEnter={() => setShow(true)}
            onMouseLeave={() => setShow(false)}
        >
            <span className="info-icon-small">ℹ️</span>
            {show && (
                <div className="tooltip-popup">
                    <p style={{ whiteSpace: "pre-line" }}>
                        {trustInfo && tooltipKey === "trustScore" ? getTrustContent() : content}
                    </p>
                </div>
            )}
        </span>
    );
}

export default InfoTooltip;
export { TOOLTIP_TEXT };
