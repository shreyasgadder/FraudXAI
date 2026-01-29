import { useState } from "react";
import InfoTooltip from "./InfoTooltip";

/**
 * FeaturesImportance - Feature Contribution Analysis
 * Displays top 8 features with their importance values
 */
function FeaturesImportance({ features }) {
    const [showAll, setShowAll] = useState(false);

    if (!features || features.length === 0) return null;

    // Show top 8 or all features based on toggle
    const displayFeatures = showAll ? features : features.slice(0, 8);

    // Format feature name (Hour: 8 -> Hour: 08:00)
    const formatFeatureName = (name) => {
        if (typeof name === 'string' && name.startsWith("Hour:")) {
            const parts = name.split(":");
            if (parts.length > 1) {
                const hour = parseInt(parts[1].trim());
                if (!isNaN(hour)) {
                    return `Hour: ${hour.toString().padStart(2, '0')}:00`;
                }
            }
        }
        return name;
    };

    return (
        <div className="drivers-panel" style={{ marginTop: '1.5rem' }}>
            <div className="panel-header">
                <span className="panel-title">FEATURE CONTRIBUTION ANALYSIS</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <button
                        onClick={() => setShowAll(!showAll)}
                        style={{
                            padding: '0.25rem 0.5rem',
                            fontSize: '0.7rem',
                            background: 'var(--bg-dark)',
                            border: '1px solid var(--border-color)',
                            borderRadius: '4px',
                            color: 'var(--text-secondary)',
                            cursor: 'pointer',
                            transition: 'all 0.2s'
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.background = 'rgba(255,255,255,0.1)';
                            e.currentTarget.style.color = 'var(--text-primary)';
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.background = 'var(--bg-dark)';
                            e.currentTarget.style.color = 'var(--text-secondary)';
                        }}
                        title={showAll ? "Show top 8 features" : "Show all features"}
                    >
                        {showAll ? "Show Top 8" : "Show All"}
                    </button>
                    <InfoTooltip tooltipKey="featuresImportance" />
                </div>
            </div>

            <table className="drivers-table">
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Impact</th>
                    </tr>
                </thead>
                <tbody>
                    {displayFeatures.map((feat, idx) => {
                        const value = feat.Importance;
                        const isPositive = value > 0;

                        return (
                            <tr key={idx}>
                                <td>{formatFeatureName(feat.Feature)}</td>
                                <td className={isPositive ? "impact-positive" : "impact-negative"}>
                                    {isPositive ? "↑" : "↓"} {Math.abs(value).toFixed(2)}
                                </td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
            <div className="drivers-legend">
                ↑ increases fraud risk | ↓ decreases fraud risk • {displayFeatures.length}/{features.length} features showing
            </div>
        </div>
    );
}

export default FeaturesImportance;
