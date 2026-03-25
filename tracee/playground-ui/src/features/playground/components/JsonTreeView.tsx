import React, { useState } from 'react';
import type { FieldDeviation } from '../../../utils/schemaAggregation';

interface Props {
  data: unknown;
  deviations?: FieldDeviation[];
}

const JsonTreeView: React.FC<Props> = ({ data, deviations = [] }) => {
  const deviationPaths = new Set(deviations.map(d => d.path));

  return (
    <div className="json-tree">
      <JsonNode value={data} path="" deviationPaths={deviationPaths} deviations={deviations} />
    </div>
  );
};

interface JsonNodeProps {
  value: unknown;
  path: string;
  keyName?: string;
  deviationPaths: Set<string>;
  deviations: FieldDeviation[];
  isLast?: boolean;
}

const JsonNode: React.FC<JsonNodeProps> = ({
  value, path, keyName, deviationPaths, deviations, isLast = true,
}) => {
  const [collapsed, setCollapsed] = useState(false);
  const isDeviation = deviationPaths.has(path);
  const deviation = isDeviation ? deviations.find(d => d.path === path) : undefined;

  const comma = isLast ? '' : ',';

  if (value === null) {
    return (
      <Row path={path} isDeviation={isDeviation} deviation={deviation}>
        {keyName !== undefined && <><span className="json-tree__key">"{keyName}"</span><span className="json-tree__colon">:</span></>}
        <span className="json-tree__null">null</span>{comma}
      </Row>
    );
  }

  if (typeof value === 'string') {
    return (
      <Row path={path} isDeviation={isDeviation} deviation={deviation}>
        {keyName !== undefined && <><span className="json-tree__key">"{keyName}"</span><span className="json-tree__colon">:</span></>}
        <span className="json-tree__string">"{value}"</span>{comma}
      </Row>
    );
  }

  if (typeof value === 'number') {
    return (
      <Row path={path} isDeviation={isDeviation} deviation={deviation}>
        {keyName !== undefined && <><span className="json-tree__key">"{keyName}"</span><span className="json-tree__colon">:</span></>}
        <span className="json-tree__number">{String(value)}</span>{comma}
      </Row>
    );
  }

  if (typeof value === 'boolean') {
    return (
      <Row path={path} isDeviation={isDeviation} deviation={deviation}>
        {keyName !== undefined && <><span className="json-tree__key">"{keyName}"</span><span className="json-tree__colon">:</span></>}
        <span className="json-tree__boolean">{String(value)}</span>{comma}
      </Row>
    );
  }

  if (Array.isArray(value)) {
    const isEmpty = value.length === 0;
    return (
      <div>
        <Row path={path} isDeviation={isDeviation} deviation={deviation}>
          <button className="json-tree__toggle" onClick={() => setCollapsed(!collapsed)}>
            {collapsed ? '▶' : '▼'}
          </button>
          {keyName !== undefined && <><span className="json-tree__key">"{keyName}"</span><span className="json-tree__colon">:</span></>}
          <span className="json-tree__bracket">[</span>
          {(collapsed || isEmpty) && <span className="json-tree__bracket">{isEmpty ? ']' : `...] (${value.length})`}</span>}
          {(collapsed || isEmpty) && comma}
        </Row>
        {!collapsed && !isEmpty && (
          <div className="json-tree__indent">
            {value.map((item, i) => (
              <JsonNode
                key={i}
                value={item}
                path={`${path}[${i}]`}
                deviationPaths={deviationPaths}
                deviations={deviations}
                isLast={i === value.length - 1}
              />
            ))}
            <Row path=""><span className="json-tree__bracket">]</span>{comma}</Row>
          </div>
        )}
      </div>
    );
  }

  if (typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>);
    const isEmpty = entries.length === 0;
    return (
      <div>
        <Row path={path} isDeviation={isDeviation} deviation={deviation}>
          <button className="json-tree__toggle" onClick={() => setCollapsed(!collapsed)}>
            {collapsed ? '▶' : '▼'}
          </button>
          {keyName !== undefined && <><span className="json-tree__key">"{keyName}"</span><span className="json-tree__colon">:</span></>}
          <span className="json-tree__bracket">{'{'}</span>
          {(collapsed || isEmpty) && <span className="json-tree__bracket">{isEmpty ? '}' : `...} (${entries.length})`}</span>}
          {(collapsed || isEmpty) && comma}
        </Row>
        {!collapsed && !isEmpty && (
          <div className="json-tree__indent">
            {entries.map(([key, val], i) => (
              <JsonNode
                key={key}
                value={val}
                keyName={key}
                path={path ? `${path}.${key}` : key}
                deviationPaths={deviationPaths}
                deviations={deviations}
                isLast={i === entries.length - 1}
              />
            ))}
            <Row path=""><span className="json-tree__bracket">{'}'}</span>{comma}</Row>
          </div>
        )}
      </div>
    );
  }

  return (
    <Row path={path} isDeviation={isDeviation} deviation={deviation}>
      {keyName !== undefined && <><span className="json-tree__key">"{keyName}"</span><span className="json-tree__colon">:</span></>}
      <span>{String(value)}</span>{comma}
    </Row>
  );
};

interface RowProps {
  path: string;
  isDeviation?: boolean;
  deviation?: FieldDeviation;
  children: React.ReactNode;
}

const Row: React.FC<RowProps> = ({ path, isDeviation, deviation, children }) => {
  const className = `json-tree__row${isDeviation ? ' json-tree__row--deviation' : ''}`;
  const title = deviation
    ? `${deviation.type}${deviation.expected ? ` (expected: ${deviation.expected})` : ''}${deviation.actual ? ` (got: ${deviation.actual})` : ''}`
    : undefined;

  return (
    <div className={className} title={title}>
      {children}
      {path && <span className="json-tree__path">{path}</span>}
    </div>
  );
};

export default JsonTreeView;
