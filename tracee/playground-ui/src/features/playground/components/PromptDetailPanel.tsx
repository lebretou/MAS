import React, { useEffect, useMemo, useState } from 'react';
import { promptAPI } from '../../../services/api';
import type { PromptWithVersions } from '../../../types/prompt';

interface Props {
  data: PromptWithVersions;
  onClose: () => void;
  onMetadataUpdated: () => void;
}

const PromptDetailPanel: React.FC<Props> = ({
  data,
  onClose,
  onMetadataUpdated,
}) => {
  const { prompt, versions } = data;

  const [editingName, setEditingName] = useState(false);
  const [editingDesc, setEditingDesc] = useState(false);
  const [nameValue, setNameValue] = useState(prompt.name);
  const [descValue, setDescValue] = useState(prompt.description ?? '');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setNameValue(prompt.name);
    setDescValue(prompt.description ?? '');
  }, [prompt.name, prompt.description]);

  const latestVersion = useMemo(() => {
    if (versions.length === 0) return null;
    return versions.reduce((a, b) => a.created_at > b.created_at ? a : b);
  }, [versions]);

  const enabledCount = latestVersion
    ? latestVersion.components.filter(c => c.enabled).length
    : 0;

  const handleSaveName = async () => {
    if (!nameValue.trim() || nameValue === prompt.name) {
      setEditingName(false);
      setNameValue(prompt.name);
      return;
    }
    setSaving(true);
    await promptAPI.updatePrompt(prompt.prompt_id, {
      name: nameValue.trim(),
      description: prompt.description,
    }).catch(() => null);
    setSaving(false);
    setEditingName(false);
    onMetadataUpdated();
  };

  const handleSaveDesc = async () => {
    const newDesc = descValue.trim() || null;
    if (newDesc === (prompt.description ?? null)) {
      setEditingDesc(false);
      setDescValue(prompt.description ?? '');
      return;
    }
    setSaving(true);
    await promptAPI.updatePrompt(prompt.prompt_id, {
      name: prompt.name,
      description: newDesc,
    }).catch(() => null);
    setSaving(false);
    setEditingDesc(false);
    onMetadataUpdated();
  };

  const fmtDate = (iso: string) => {
    const d = new Date(iso);
    return d.toLocaleDateString('en-US', {
      month: 'short', day: 'numeric', year: 'numeric',
    });
  };

  return (
    <div className="prompt-detail">
      {/* header */}
      <div className="prompt-detail__header">
        <div className="prompt-detail__header-main">
          {editingName ? (
            <input
              className="input prompt-detail__name-input"
              value={nameValue}
              onChange={e => setNameValue(e.target.value)}
              onBlur={handleSaveName}
              onKeyDown={e => {
                if (e.key === 'Enter') handleSaveName();
                if (e.key === 'Escape') { setEditingName(false); setNameValue(prompt.name); }
              }}
              disabled={saving}
              autoFocus
            />
          ) : (
            <h3
              className="prompt-detail__name"
              role="button"
              tabIndex={0}
              title="click to rename"
              onClick={() => setEditingName(true)}
              onKeyDown={e => { if (e.key === 'Enter') setEditingName(true); }}
            >
              {prompt.name}
            </h3>
          )}
          <span className="prompt-detail__id">{prompt.prompt_id}</span>
        </div>
        <button
          type="button"
          className="icon-btn icon-btn--close"
          onClick={onClose}
          aria-label="close detail"
        >
          ×
        </button>
      </div>

      {/* description */}
      <div className="prompt-detail__desc-section">
        {editingDesc ? (
          <textarea
            className="textarea prompt-detail__desc-input"
            value={descValue}
            rows={2}
            placeholder="Add a description..."
            onChange={e => setDescValue(e.target.value)}
            onBlur={handleSaveDesc}
            onKeyDown={e => {
              if (e.key === 'Escape') { setEditingDesc(false); setDescValue(prompt.description ?? ''); }
            }}
            disabled={saving}
            autoFocus
          />
        ) : (
          <div
            className="prompt-detail__desc"
            role="button"
            tabIndex={0}
            title="click to edit description"
            onClick={() => setEditingDesc(true)}
            onKeyDown={e => { if (e.key === 'Enter') setEditingDesc(true); }}
          >
            {prompt.description || <span className="prompt-detail__desc-placeholder">Add a description...</span>}
          </div>
        )}
      </div>

      {/* stats row */}
      <div className="prompt-detail__stats">
        <div className="prompt-detail__stat">
          <span className="prompt-detail__stat-value">{versions.length}</span>
          <span className="prompt-detail__stat-label">versions</span>
        </div>
        <div className="prompt-detail__stat">
          <span className="prompt-detail__stat-value">{enabledCount}</span>
          <span className="prompt-detail__stat-label">components</span>
        </div>
        <div className="prompt-detail__stat">
          <span className="prompt-detail__stat-value">{fmtDate(prompt.created_at)}</span>
          <span className="prompt-detail__stat-label">created</span>
        </div>
        <div className="prompt-detail__stat">
          <span className="prompt-detail__stat-value">{fmtDate(prompt.updated_at)}</span>
          <span className="prompt-detail__stat-label">updated</span>
        </div>
      </div>

    </div>
  );
};

export default PromptDetailPanel;
