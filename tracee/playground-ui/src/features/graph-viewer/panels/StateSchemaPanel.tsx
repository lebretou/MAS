import { useMemo, useState } from "react";
import { Panel } from "@xyflow/react";
import type { ExecutionFrame } from "../../../types/node-data";
import type { JsonSchema } from "../../../types/schema";
import dictIcon from "../../../assets/dict.svg";
import listIcon from "../../../assets/list.svg";
import numberIcon from "../../../assets/number.svg";
import stringIcon from "../../../assets/string.svg";
import iconStateVariable from "../../../assets/icon-statevariable.svg";
import iconSort from "../../../assets/icon-sort.svg";

interface Props {
  schema: JsonSchema;
  activeFrame?: ExecutionFrame | null;
}

const TYPE_ICONS: Record<string, string> = {
  object: dictIcon,
  array: listIcon,
  number: numberIcon,
  string: stringIcon,
  boolean: stringIcon,
};

function unescapeNewlines(s: string): string {
  return s.replace(/\\n/g, "\n").replace(/\\t/g, "\t").replace(/\\r/g, "\r");
}

function formatStateValue(value: unknown): string {
  if (value == null) return "null";
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
      try {
        const parsed = JSON.parse(trimmed) as unknown;
        return JSON.stringify(parsed, null, 2);
      } catch {
        return unescapeNewlines(value);
      }
    }
    return unescapeNewlines(value);
  }
  return JSON.stringify(value, null, 2);
}

function inferTypeFromValue(value: unknown): string {
  if (Array.isArray(value)) return "array";
  if (value === null) return "null";
  return typeof value;
}

const STATUS_ORDER = { changed: 0, filled: 1, empty: 2 } as const;

function getStatusRank(
  key: string,
  stateSnapshot: Record<string, unknown>,
  changedKeys: Set<string>,
  canShowChanges: boolean,
): number {
  const hasSnapshotValue = key in stateSnapshot;
  const currentValue = stateSnapshot[key];
  const isChanged = canShowChanges && changedKeys.has(key);
  const isEmpty =
    !hasSnapshotValue ||
    currentValue === "" ||
    currentValue == null ||
    (Array.isArray(currentValue) && currentValue.length === 0) ||
    (typeof currentValue === "object" &&
      currentValue !== null &&
      !Array.isArray(currentValue) &&
      Object.keys(currentValue).length === 0);
  const status = isChanged ? "changed" : isEmpty ? "empty" : "filled";
  return STATUS_ORDER[status];
}

export function StateSchemaPanel({ schema, activeFrame }: Props) {
  const properties = schema.properties || {};
  const [expandedKeys, setExpandedKeys] = useState<Record<string, boolean>>({});
  const [sortByStatus, setSortByStatus] = useState(false);
  const stateSnapshot = activeFrame?.stateSnapshot ?? {};
  const changedKeys = useMemo(() => new Set(activeFrame?.changedKeys ?? []), [activeFrame]);
  const canShowChanges = Boolean(activeFrame && activeFrame.index > 0);
  const baseKeys = useMemo(
    () =>
      Array.from(
        new Set([...Object.keys(properties), ...Object.keys(stateSnapshot)]),
      ).sort(),
    [properties, stateSnapshot],
  );
  const rowKeys = useMemo(() => {
    if (!sortByStatus) return baseKeys;
    return [...baseKeys].sort((a, b) => {
      const rankA = getStatusRank(a, stateSnapshot, changedKeys, canShowChanges);
      const rankB = getStatusRank(b, stateSnapshot, changedKeys, canShowChanges);
      return rankA !== rankB ? rankA - rankB : a.localeCompare(b);
    });
  }, [baseKeys, sortByStatus, stateSnapshot, changedKeys, canShowChanges]);

  return (
    <Panel position="top-right" className="state-schema-panel">
      <div className="state-schema-panel__header">
        <h3 className="state-schema-panel__title">
          <img src={iconStateVariable} alt="" className="state-schema-panel__title-icon" aria-hidden />
          State Variables
        </h3>
        <p className="state-schema-panel__subtitle">
          {activeFrame ? `${activeFrame.label} · ${activeFrame.changedKeys.length} changed` : "Schema overview"}
        </p>
      </div>
      <div className="state-schema-panel__table-header">
        <div className="state-schema-panel__col-name">Name</div>
        <div className="state-schema-panel__col-type">Type</div>
        <div className="state-schema-panel__col-status">
          Status
          <button
            type="button"
            className={`state-schema-panel__sort-btn${sortByStatus ? " is-active" : ""}`}
            onClick={() => setSortByStatus((prev) => !prev)}
            title={sortByStatus ? "Sort by name" : "Sort by status (changed → filled → empty)"}
            aria-pressed={sortByStatus}
          >
            <img src={iconSort} alt="" aria-hidden />
          </button>
        </div>
      </div>
      <div className="state-schema-panel__content">
        {rowKeys.map((key) => {
          const propSchema = properties[key];
          const hasSnapshotValue = key in stateSnapshot;
          const currentValue = stateSnapshot[key];
          const typeStr = Array.isArray(propSchema?.type)
            ? propSchema.type[0]
            : propSchema?.type || inferTypeFromValue(currentValue);
          const displayTypeStr = Array.isArray(propSchema?.type)
            ? propSchema.type.join(" | ")
            : propSchema?.type || inferTypeFromValue(currentValue);
          const iconSrc = TYPE_ICONS[typeStr] || stringIcon;
          const isChanged = canShowChanges && changedKeys.has(key);
          const isEmpty = !hasSnapshotValue
            || currentValue === ""
            || currentValue == null
            || (Array.isArray(currentValue) && currentValue.length === 0)
            || (typeof currentValue === "object" && currentValue !== null && !Array.isArray(currentValue) && Object.keys(currentValue).length === 0);
          const isExpanded = Boolean(expandedKeys[key]);
          const canExpand = hasSnapshotValue;

          return (
            <div key={key} className="state-schema-panel__row-container">
              <div
                className={`state-schema-panel__row${isChanged ? " state-schema-panel__row--changed" : ""}${isExpanded ? " is-expanded" : ""}`}
                onClick={() => setExpandedKeys((prev) => ({ ...prev, [key]: !prev[key] }))}
              >
                <div className="state-schema-panel__col-name state-schema-panel__key">{key}</div>
                <div className="state-schema-panel__col-type state-schema-panel__type-meta">
                  <img src={iconSrc} alt={typeStr} className="state-schema-panel__type-icon" />
                  <span>{displayTypeStr}</span>
                </div>
                <div className="state-schema-panel__col-status">
                  <span
                    className={`state-schema-panel__status${isChanged ? " is-changed" : isEmpty ? " is-empty" : " is-filled"}`}
                  >
                    {isChanged ? "changed" : isEmpty ? "empty" : "filled"}
                  </span>
                </div>
              </div>
              {isExpanded && (
                <div className="state-schema-panel__value-container">
                  {isEmpty ? (
                    <span className="state-schema-panel__value-empty">empty</span>
                  ) : (
                    <pre className="state-schema-panel__value">{formatStateValue(currentValue)}</pre>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </Panel>
  );
}
