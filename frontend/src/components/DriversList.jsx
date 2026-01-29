function DriversList({ features, type }) {
    if (!features || features.length === 0) return null;

    const topFeatures = features.slice(0, 5);

    return (
        <div className="drivers-panel">
            <table className="drivers-table">
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Impact</th>
                    </tr>
                </thead>
                <tbody>
                    {topFeatures.map((feat, idx) => {
                        const value = type === "shap" ? feat.SHAP_Value : feat.GNN_Value;
                        const isPositive = value > 0;

                        return (
                            <tr key={idx}>
                                <td>{feat.Feature}</td>
                                <td className={isPositive ? "impact-positive" : "impact-negative"}>
                                    {isPositive ? "↑" : "↓"} {Math.abs(value).toFixed(2)}
                                </td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
            <div className="drivers-legend">
                ↑ increases fraud risk &nbsp;|&nbsp; ↓ decreases fraud risk
            </div>
        </div>
    );
}

export default DriversList;
