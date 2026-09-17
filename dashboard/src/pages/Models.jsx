import { useState, useEffect } from 'react'
import { useApp } from '../App'

const PATTERN_DESCRIPTIONS = {
  'FAN-OUT': 'One account sends to many different accounts within a short time window.',
  'FAN-IN': 'Many accounts all send to a single destination account.',
  'CYCLE': 'Money cycles through a ring of accounts and eventually returns to origin.',
  'STACK': 'Multiple parallel 2-hop chains that layer transactions to obscure flow.',
  'RANDOM': 'Multi-hop random walk chain — each hop goes to a new account to obscure trail.',
  'BIPARTITE': 'Two structured groups of accounts exchanging in a structured bipartite pattern.',
  'GATHER-SCATTER': 'Fan-in to a central collector account, then fan-out to many recipients.',
  'SCATTER-GATHER': 'Fan-out from one account, then all recipients consolidate to a single collector.',
}

export default function ModelsPage() {
  const { API_URL } = useApp()
  const [health, setHealth] = useState(null)
  const [detectorStats, setDetectorStats] = useState(null)
  const [weights, setWeights] = useState({ rule_weight: 0.20, xgb_weight: 0.35, gnn_weight: 0.45 })
  const [threshold, setThreshold] = useState(0.50)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    const load = async () => {
      try {
        const [h, s] = await Promise.all([
          fetch(`${API_URL}/api/health`).then(r => r.json()),
          fetch(`${API_URL}/api/detector/stats`).then(r => r.json()),
        ])
        setHealth(h)
        setDetectorStats(s)
      } catch {}
    }
    load()
    const interval = setInterval(load, 5000)
    return () => clearInterval(interval)
  }, [API_URL])

  const saveWeights = async () => {
    setSaving(true)
    try {
      const total = weights.rule_weight + weights.xgb_weight + weights.gnn_weight
      await fetch(`${API_URL}/api/detector/weights`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rule_weight: weights.rule_weight / total,
          xgb_weight: weights.xgb_weight / total,
          gnn_weight: weights.gnn_weight / total,
          threshold,
        }),
      })
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } catch {}
    setSaving(false)
  }

  const modelItems = [
    {
      key: 'rule_engine', name: 'Rule Engine', icon: '📐',
      desc: 'Expert-crafted heuristics for all 8 laundering patterns.',
      status: true, // always available
      color: '#f59e0b',
      details: ['8 pattern detectors', 'Graph cycle detection', 'Amount similarity analysis', 'Fan-in/out thresholds'],
    },
    {
      key: 'xgboost', name: 'XGBoost', icon: '🌲',
      desc: 'Gradient-boosted tree model trained with SMOTE on 5M+ transactions.',
      status: health?.models_loaded?.xgboost,
      color: '#3b82f6',
      details: ['500 estimators', 'CUDA tree_method=hist', 'SHAP explainability', 'Threshold-tuned (F1-optimal)'],
    },
    {
      key: 'gnn', name: 'Graph Transformer', icon: '🕸',
      desc: 'PyTorch Geometric transformer with multi-head attention on transaction graph.',
      status: health?.models_loaded?.gnn,
      color: '#8b5cf6',
      details: ['3-layer Graph Transformer', '4-head attention', 'Focal Loss for imbalance', 'CUDA-accelerated'],
    },
  ]

  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-title">Model Configuration</div>
          <div className="page-subtitle">Ensemble weights, thresholds, and model health</div>
        </div>
      </div>

      <div className="page-body">

        {/* Detector Stats */}
        {detectorStats && (
          <div className="kpi-grid">
            {[
              { label: 'Processed', value: detectorStats.processed || 0, icon: '⚡', color: 'blue' },
              { label: 'Alerted', value: detectorStats.alerted || 0, icon: '🚨', color: 'red' },
              { label: 'Retroactive', value: detectorStats.retroactive_flagged || 0, icon: '↩️', color: 'purple' },
              { label: 'Suppressed', value: detectorStats.suppressed || 0, icon: '🔇', color: 'green' },
            ].map(item => (
              <div key={item.label} className={`kpi-card ${item.color}`}>
                <div className="kpi-icon">{item.icon}</div>
                <div className="kpi-value">{item.value.toLocaleString()}</div>
                <div className="kpi-label">{item.label}</div>
              </div>
            ))}
          </div>
        )}

        {/* Model Status Cards */}
        <div className="grid-3">
          {modelItems.map(m => (
            <div key={m.key} className="card" style={{ position: 'relative', overflow: 'hidden' }}>
              <div style={{
                position: 'absolute', top: 0, left: 0, right: 0, height: 3,
                background: m.status ? m.color : 'var(--text-muted)'
              }} />
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                <span style={{ fontSize: 24 }}>{m.icon}</span>
                <div>
                  <div style={{ fontWeight: 700, fontSize: 14 }}>{m.name}</div>
                  <div style={{
                    fontSize: 11, fontWeight: 600,
                    color: m.status ? 'var(--green)' : 'var(--text-muted)'
                  }}>
                    {m.status === undefined ? '⏳ Checking...' : m.status ? '✓ Loaded' : '✗ Not loaded — run train.py'}
                  </div>
                </div>
              </div>
              <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 12, lineHeight: 1.5 }}>{m.desc}</p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                {m.details.map(d => (
                  <div key={d} style={{ fontSize: 11, color: 'var(--text-muted)', display: 'flex', gap: 6 }}>
                    <span style={{ color: m.color }}>›</span> {d}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Ensemble Weights Configuration */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">⚖️ Ensemble Weights & Threshold</span>
            <button className="btn btn-primary btn-sm" onClick={saveWeights} disabled={saving}>
              {saved ? '✓ Saved!' : saving ? '...' : 'Apply Changes'}
            </button>
          </div>

          <div className="grid-3" style={{ marginBottom: 20 }}>
            {[
              { key: 'rule_weight', label: 'Rule Engine Weight', icon: '📐', color: '#f59e0b' },
              { key: 'xgb_weight', label: 'XGBoost Weight', icon: '🌲', color: '#3b82f6' },
              { key: 'gnn_weight', label: 'GNN Weight', icon: '🕸', color: '#8b5cf6' },
            ].map(item => (
              <div key={item.key} style={{ background: 'var(--bg-secondary)', padding: '14px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
                  <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{item.icon} {item.label}</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 14, fontWeight: 700, color: item.color }}>
                    {Math.round(weights[item.key] * 100)}%
                  </span>
                </div>
                <input type="range" min="0" max="100" step="5"
                  value={Math.round(weights[item.key] * 100)}
                  onChange={e => setWeights(w => ({ ...w, [item.key]: parseInt(e.target.value) / 100 }))}
                  style={{ width: '100%', accentColor: item.color }}
                />
              </div>
            ))}
          </div>

          <div style={{ background: 'var(--bg-secondary)', padding: '14px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
              <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>🎯 Decision Threshold</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 14, fontWeight: 700, color: threshold > 0.7 ? 'var(--green)' : threshold > 0.4 ? 'var(--yellow)' : 'var(--red)' }}>
                {Math.round(threshold * 100)}%
              </span>
            </div>
            <input type="range" min="0" max="100" step="1"
              value={Math.round(threshold * 100)}
              onChange={e => setThreshold(parseInt(e.target.value) / 100)}
              style={{ width: '100%', accentColor: 'var(--blue)' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
              <span>← More Sensitive (more alerts)</span>
              <span>More Specific (fewer alerts) →</span>
            </div>
          </div>
        </div>

        {/* Pattern Reference */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">📚 Laundering Pattern Reference</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            {Object.entries(PATTERN_DESCRIPTIONS).map(([pattern, desc]) => (
              <div key={pattern} style={{
                padding: '12px 14px', background: 'var(--bg-secondary)',
                borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)',
                display: 'flex', gap: 12, alignItems: 'flex-start'
              }}>
                <span className={`pattern-badge ${pattern}`}>{pattern}</span>
                <p style={{ fontSize: 12, color: 'var(--text-secondary)', margin: 0, lineHeight: 1.5 }}>{desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Training Command Reference */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">🖥️ Quick Commands</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {[
              { label: 'Train all models (full dataset)', cmd: '.\\aml_env\\Scripts\\python train.py' },
              { label: 'Quick test (500K rows)', cmd: '.\\aml_env\\Scripts\\python train.py --nrows 500000' },
              { label: 'Train without GNN (CPU-only)', cmd: '.\\aml_env\\Scripts\\python train.py --skip-gnn' },
              { label: 'Start API server', cmd: '.\\aml_env\\Scripts\\python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload' },
              { label: 'Simulate 10K transactions', cmd: '.\\aml_env\\Scripts\\python simulate_stream.py --max-txs 10000 --speed 500' },
              { label: 'Simulate laundering only (for testing)', cmd: '.\\aml_env\\Scripts\\python simulate_stream.py --include-laundering-only --speed 10' },
            ].map(item => (
              <div key={item.label} style={{
                display: 'flex', alignItems: 'center', gap: 12, padding: '10px 14px',
                background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)'
              }}>
                <span style={{ fontSize: 12, color: 'var(--text-secondary)', flex: '0 0 220px' }}>{item.label}</span>
                <code style={{
                  fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--cyan)',
                  background: 'rgba(6,182,212,0.05)', padding: '4px 10px',
                  borderRadius: 4, border: '1px solid rgba(6,182,212,0.15)', flex: 1
                }}>
                  {item.cmd}
                </code>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
