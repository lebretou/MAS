import { Fragment } from "react";
import iconTool from "../assets/icon-tool.svg";
import iconState from "../assets/icon-state.svg";
import iconError from "../assets/icon-error.svg";
import iconCognition from "../assets/cognition.svg";

type ChipType = "tool" | "state" | "agent" | "error" | "warning";

interface Chip {
  type: ChipType;
  value: string;
}

interface TextSegment {
  kind: "text";
  value: string;
}

interface ChipSegment {
  kind: "chip";
  chip: Chip;
}

type Segment = TextSegment | ChipSegment;

const TAG_REGEX = /\{(tool|state|agent|error|warning):([^}]+)\}/g;

const CHIP_CONFIG: Record<ChipType, { icon: string; className: string }> = {
  tool: { icon: iconTool, className: "cognition-chip--tool" },
  state: { icon: iconState, className: "cognition-chip--state" },
  agent: { icon: iconCognition, className: "cognition-chip--agent" },
  error: { icon: iconError, className: "cognition-chip--error" },
  warning: { icon: iconError, className: "cognition-chip--warning" },
};

function parseSegments(text: string): Segment[] {
  const segments: Segment[] = [];
  let lastIndex = 0;

  for (const match of text.matchAll(TAG_REGEX)) {
    const matchIndex = match.index!;
    if (matchIndex > lastIndex) {
      segments.push({ kind: "text", value: text.slice(lastIndex, matchIndex) });
    }
    segments.push({
      kind: "chip",
      chip: { type: match[1] as ChipType, value: match[2] },
    });
    lastIndex = matchIndex + match[0].length;
  }

  if (lastIndex < text.length) {
    segments.push({ kind: "text", value: text.slice(lastIndex) });
  }

  return segments;
}

interface Props {
  text: string;
  onAgentClick?: (agentId: string) => void;
  onToolClick?: (value: string) => void;
  onStateClick?: (value: string) => void;
}

export function CognitionText({ text, onAgentClick, onToolClick, onStateClick }: Props) {
  const segments = parseSegments(text);

  return (
    <span className="cognition-text">
      {segments.map((seg, i) => {
        if (seg.kind === "text") {
          return <Fragment key={i}>{seg.value}</Fragment>;
        }
        const config = CHIP_CONFIG[seg.chip.type];
        const clickHandler =
          seg.chip.type === "agent" && onAgentClick ? () => onAgentClick(seg.chip.value) :
          seg.chip.type === "tool" && onToolClick ? () => onToolClick(seg.chip.value) :
          seg.chip.type === "state" && onStateClick ? () => onStateClick(seg.chip.value) :
          undefined;
        const isClickable = Boolean(clickHandler);
        return (
          <span
            key={i}
            className={`cognition-chip ${config.className}${isClickable ? " cognition-chip--clickable" : ""}`}
            onClick={clickHandler}
            role={isClickable ? "button" : undefined}
            tabIndex={isClickable ? 0 : undefined}
          >
            <img src={config.icon} alt="" className="cognition-chip__icon" />
            <span className="cognition-chip__label">{seg.chip.value}</span>
          </span>
        );
      })}
    </span>
  );
}
