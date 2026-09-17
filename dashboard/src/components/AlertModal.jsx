import { useState, useRef, useEffect } from 'react'
import * as d3 from 'd3'

// ------------------------------------------------------------------ //
//  D3 Graph Visualization Component                                   //
// ------------------------------------------------------------------ //
function GraphView({ alert }) {
  const svgRef = useRef(null)

  useEffect(() => {
    if (!svgRef.current || !alert) return

    const involved = alert.involved_accounts || [alert.from_account, alert.to_account]
    const nodes = [...new Set([alert.from_account, alert.to_account, ...involved])]
      .filter(Boolean)
      .map(id => ({ id, isSender: id === alert.from_account, isReceiver: id === alert.to_account }))

    const links = []
    // Primary transaction link
    if (alert.from_account && alert.to_account) {
      links.push({ source: alert.from_account, target: alert.to_account, primary: true })
    }
    // Involved account links
    involved.forEach(acc => {
      if (acc !== alert.from_account && acc !== alert.to_account) {
        links.push({ source: alert.from_account, target: acc, primary: false })
      }
    })

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const width = svgRef.current.clientWidth || 460
    const height = 240

    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(80))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))

    const g = svg.append('g')

    // Arrow marker
    svg.append('defs').append('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#3b82f6')

    const link = g.append('g').selectAll('line')
      .data(links).join('line')
      .attr('stroke', d => d.primary ? '#3b82f6' : 'rgba(139,92,246,0.5)')
      .attr('stroke-width', d => d.primary ? 2 : 1)
      .attr('stroke-dasharray', d => d.primary ? null : '4,3')
      .attr('marker-end', 'url(#arrow)')

    const node = g.append('g').selectAll('g')
      .data(nodes).join('g')
      .call(d3.drag()
        .on('start', (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y })
        .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y })
        .on('end', (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null })
      )

    node.append('circle')
      .attr('r', d => d.isSender ? 18 : d.isReceiver ? 16 : 12)
      .attr('fill', d => d.isSender ? 'rgba(239,68,68,0.2)' : d.isReceiver ? 'rgba(59,130,246,0.2)' : 'rgba(139,92,246,0.2)')
      .attr('stroke', d => d.isSender ? '#ef4444' : d.isReceiver ? '#3b82f6' : '#8b5cf6')
      .attr('stroke-width', 1.5)

    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('fill', '#f0f4ff')
      .attr('font-size', '9px')
      .attr('font-family', 'JetBrains Mono, monospace')
      .text(d => (d.id || '').slice(0, 8))

    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y)
      node.attr('transform', d => `translate(${d.x},${d.y})`)
    })

    return () => simulation.stop()
  }, [alert])

  return (
    <div className="graph-container">
      <svg ref={svgRef} style={{ width: '100%', height: 240 }} />
      <div style={{ position: 'absolute', bottom: 8, right: 12, display: 'flex', gap: 12, fontSize: 10, color: 'var(--text-muted)' }}>
        <span>🔴 Sender</span>
        <span>🔵 Receiver</span>
        <span>🟣 Linked</span>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ //
//  SHAP Explainer                                                      //
// ------------------------------------------------------------------ //
function SHAPExplainer({ shapValues }) {
  if (!shapValues || Object.keys(shapValues).length === 0) {
    return <div style={{ color: 'var(--text-muted)', fontSize: 12, textAlign: 'center', padding: 20 }}>
      Train the XGBoost model to see SHAP explanations.
    </div>
  }

  const sorted = Object.entries(shapValues)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 10)

  const maxAbs = Math.max(...sorted.map(([, v]) => Math.abs(v)), 0.001)

  return (
    <div>
      {sorted.map(([feat, val]) => (
        <div key={feat} className="shap-bar-wrap">
          <div className="shap-feature-name">{feat}</div>
          <div className="shap-bar">
            <div className="shap-fill" style={{
              width: `${Math.min(100, (Math.abs(val) / maxAbs) * 100)}%`,
              marginLeft: val < 0 ? 0 : undefined,
            }} />
          </div>
          <div className={`shap-fill ${val >= 0 ? 'positive' : 'negative'}`}
            style={{ width: `${Math.min(100, (Math.abs(val) / maxAbs) * 100)}%` }} />
          <div className="shap-value">{val >= 0 ? '+' : ''}{val.toFixed(3)}</div>
        </div>
      ))}
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8 }}>
        🔴 Increases laundering score &nbsp;|&nbsp; 🟢 Decreases laundering score
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ //
//  Model Score Breakdown                                               //
// ------------------------------------------------------------------ //
function ModelScores({ scores, finalScore, pattern }) {
  const items = [
    { name: 'Rule Engine', key: 'rule', icon: '📐', color: '#f59e0b', weight: '20%' },
    { name: 'XGBoost', key: 'xgboost', icon: '🌲', color: '#3b82f6', weight: '35%' },
    { name: 'Graph NN', key: 'gnn', icon: '🕸', color: '#8b5cf6', weight: '45%' },
  ]

  return (
    <div>
      <div className="model-scores">
        {items.map(item => {
          const score = scores?.[item.key] ?? 0
          return (
            <div key={item.key} className="model-score-item">
              <div className="model-name">{item.icon} {item.name}</div>
              <div className="model-value" style={{ color: score > 0.5 ? '#ef4444' : '#10b981' }}>
                {Math.round(score * 100)}%
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
                weight {item.weight}
              </div>
              <div className="score-bar" style={{ marginTop: 6 }}>
                <div className="score-fill" style={{
                  width: `${Math.round(score * 100)}%`,
                  background: score > 0.7 ? item.color : 'rgba(255,255,255,0.2)'
                }} />
              </div>
            </div>
          )
        })}
      </div>

      <div style={{
        marginTop: 16, padding: '12px 16px',
        background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)',
        border: '1px solid var(--border-accent)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between'
      }}>
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>ENSEMBLE FINAL SCORE</div>
          <div style={{ fontSize: 28, fontWeight: 800, color: (finalScore || 0) > 0.5 ? '#ef4444' : '#10b981', fontFamily: 'var(--font-mono)' }}>
            {Math.round((finalScore || 0) * 100)}%
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>DETECTED PATTERN</div>
          <span className={`pattern-badge ${pattern}`} style={{ fontSize: 13 }}>{pattern || 'UNKNOWN'}</span>
        </div>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ //
//  Main Alert Modal                                                    //
// ------------------------------------------------------------------ //
export default function AlertModal({ alert, onClose, onReview }) {
  const [note, setNote] = useState('')
  const [whitelistAccounts, setWhitelistAccounts] = useState(
    (alert.involved_accounts || [alert.from_account, alert.to_account]).filter(Boolean)
  )
  const [ttl, setTtl] = useState(30)
  const [activeTab, setActiveTab] = useState('overview')
  const [loading, setLoading] = useState(false)

  const handleReview = async (action) => {
    setLoading(true)
    await onReview(action, note, action === 'APPROVED' ? whitelistAccounts : [])
    setLoading(false)
  }

  const tabs = ['overview', 'graph', 'models', 'explain']

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span className={`severity-badge ${alert.severity}`}>{alert.severity}</span>
            <div className="modal-title">
              Alert Detail
              {alert.retroactive && <span className="retro-tag" style={{ marginLeft: 8 }}>↩ RETROACTIVE</span>}
            </div>
          </div>
          <button className="btn-icon" onClick={onClose}>✕</button>
        </div>

        <div className="modal-body">
          {/* Quick Info */}
          <div style={{
            display: 'grid', gridTemplateColumns: '1fr 1fr 1fr',
            gap: 12, marginBottom: 20
          }}>
            {[
              { label: 'From Account', value: alert.from_account, mono: true },
              { label: 'To Account', value: alert.to_account, mono: true },
              { label: 'Amount', value: `$${(alert.amount_usd || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}`, mono: true },
              { label: 'Payment', value: alert.payment_format },
              { label: 'Timestamp', value: alert.timestamp ? new Date(alert.timestamp).toLocaleString() : '--' },
              { label: 'TX ID', value: (alert.tx_id || '').slice(0, 12) + '...', mono: true },
            ].map(item => (
              <div key={item.label} style={{
                background: 'var(--bg-secondary)', padding: '10px 14px',
                borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)'
              }}>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 3 }}>{item.label}</div>
                <div style={{ fontSize: 12, color: 'var(--text-primary)', fontFamily: item.mono ? 'var(--font-mono)' : undefined }}>
                  {item.value || '--'}
                </div>
              </div>
            ))}
          </div>

          {/* Tabs */}
          <div style={{ display: 'flex', gap: 4, marginBottom: 16, borderBottom: '1px solid var(--border)', paddingBottom: 0 }}>
            {tabs.map(t => (
              <button key={t} onClick={() => setActiveTab(t)} style={{
                padding: '7px 16px', background: 'none', border: 'none', cursor: 'pointer',
                color: activeTab === t ? 'var(--blue)' : 'var(--text-muted)',
                fontSize: 13, fontWeight: 600, borderBottom: activeTab === t ? '2px solid var(--blue)' : '2px solid transparent',
                fontFamily: 'inherit', textTransform: 'capitalize',
              }}>
                {t === 'overview' ? '📋 Overview' : t === 'graph' ? '🕸 Graph' : t === 'models' ? '🧠 Models' : '💡 Explain'}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          {activeTab === 'overview' && (
            <div>
              <div className="grid-2">
                <div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8 }}>Triggered Patterns</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                    {(alert.triggered_rules || [alert.detected_pattern]).filter(Boolean).map(p => (
                      <span key={p} className={`pattern-badge ${p}`}>{p}</span>
                    ))}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8 }}>Involved Accounts ({(alert.involved_accounts || []).length})</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {(alert.involved_accounts || []).slice(0, 8).map(a => (
                      <span key={a} className="account-chip">{(a || '').slice(0, 10)}</span>
                    ))}
                    {(alert.involved_accounts || []).length > 8 && (
                      <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>+{(alert.involved_accounts || []).length - 8} more</span>
                    )}
                  </div>
                </div>
              </div>

              {alert.retroactive && (
                <div style={{
                  marginTop: 12, padding: '10px 14px',
                  background: 'rgba(139,92,246,0.08)', border: '1px solid rgba(139,92,246,0.2)',
                  borderRadius: 'var(--radius-sm)', fontSize: 12, color: '#c4b5fd'
                }}>
                  ↩ <strong>Retroactively flagged:</strong> {alert.retroactive_reason}
                </div>
              )}

              {/* Entity info */}
              {(alert.from_entity_type || alert.to_entity_type) && (
                <div className="grid-2" style={{ marginTop: 12 }}>
                  <div style={{ background: 'var(--bg-secondary)', padding: '10px 14px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)' }}>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>SENDER TYPE</div>
                    <div style={{ fontSize: 13, marginTop: 4 }}>{alert.from_entity_type || 'Unknown'}</div>
                  </div>
                  <div style={{ background: 'var(--bg-secondary)', padding: '10px 14px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)' }}>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>RECEIVER TYPE</div>
                    <div style={{ fontSize: 13, marginTop: 4 }}>{alert.to_entity_type || 'Unknown'}</div>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'graph' && <GraphView alert={alert} />}

          {activeTab === 'models' && (
            <ModelScores
              scores={alert.component_scores}
              finalScore={alert.final_score}
              pattern={alert.detected_pattern}
            />
          )}

          {activeTab === 'explain' && <SHAPExplainer shapValues={alert.shap_values} />}

          {/* Review Actions */}
          {alert.status === 'PENDING' && (
            <div style={{
              marginTop: 20, padding: '16px', background: 'var(--bg-secondary)',
              borderRadius: 'var(--radius-md)', border: '1px solid var(--border)'
            }}>
              <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>Analyst Decision</div>

              <textarea
                style={{
                  width: '100%', background: 'var(--bg-input)', border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)', padding: '8px 12px', color: 'var(--text-primary)',
                  fontFamily: 'var(--font-sans)', fontSize: 13, resize: 'vertical', minHeight: 60,
                  outline: 'none',
                }}
                placeholder="Add a note (optional)..."
                value={note}
                onChange={e => setNote(e.target.value)}
              />

              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 8, marginBottom: 12 }}>
                <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Whitelist TTL:</span>
                {[7, 30, 90].map(d => (
                  <button key={d} className={`filter-chip${ttl === d ? ' active' : ''}`}
                    onClick={() => setTtl(d)} style={{ fontSize: 11 }}>
                    {d} days
                  </button>
                ))}
                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                  ({whitelistAccounts.length} accounts will be whitelisted on Approve)
                </span>
              </div>

              <div style={{ display: 'flex', gap: 10 }}>
                <button className="btn btn-success" disabled={loading} onClick={() => handleReview('APPROVED')} style={{ flex: 1, justifyContent: 'center' }}>
                  {loading ? '...' : '✓ Approve — Mark as Legitimate'}
                </button>
                <button className="btn btn-danger" disabled={loading} onClick={() => handleReview('REJECTED')} style={{ flex: 1, justifyContent: 'center' }}>
                  {loading ? '...' : '✗ Reject — Confirm Laundering'}
                </button>
              </div>
            </div>
          )}

          {alert.status !== 'PENDING' && (
            <div style={{ marginTop: 16, padding: '12px 16px', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)' }}>
              <span className={`status-badge ${alert.status}`}>{alert.status}</span>
              {alert.reviewed_at && (
                <span style={{ marginLeft: 10, fontSize: 11, color: 'var(--text-muted)' }}>
                  at {new Date(alert.reviewed_at).toLocaleString()}
                </span>
              )}
              {alert.reviewer_note && (
                <div style={{ marginTop: 6, fontSize: 12, color: 'var(--text-secondary)' }}>
                  Note: {alert.reviewer_note}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
