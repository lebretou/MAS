import React from 'react';

export interface VisualizationLegendEntry {
  id: string;
  label: string;
  tone: 'primary' | 'compare' | 'anchor';
}

interface Props {
  items: VisualizationLegendEntry[];
  className?: string;
}

const VisualizationLegend: React.FC<Props> = ({ items, className }) => {
  if (items.length === 0) {
    return null;
  }

  return (
    <div className={className ? `viz-legend ${className}` : 'viz-legend'} aria-label="Visualization legend">
      {items.map((item) => (
        <div key={item.id} className="viz-legend__item">
          <span className={`viz-legend__dot viz-legend__dot--${item.tone}`} aria-hidden />
          <span className="viz-legend__label">{item.label}</span>
        </div>
      ))}
    </div>
  );
};

export default VisualizationLegend;
