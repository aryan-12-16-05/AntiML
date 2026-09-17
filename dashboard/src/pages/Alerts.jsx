import { useState, useEffect, useRef } from 'react'
import { useApp } from '../App'
import AlertModal from '../components/AlertModal'
import * as d3 from 'd3'

const PATTERNS = ['ALL', 'FAN-OUT', 'FAN-IN', 'CYCLE', 'STACK', 'RANDOM', 'BIPARTITE', 'GATHER-SCATTER', 'SCATTER-GATHER']
const SEVERITIES = ['ALL', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
const STATUSES = ['ALL', 'PENDING', 'APPROVED', 'REJECTED']

export default function AlertsPage() {
  const { alerts, reviewAlert } = useApp()
  const [filterPattern, setFilterPattern] = useState('ALL')
  const [filterSeverity, setFilterSeverity] = useState('ALL')
  const [filterStatus, setFilterStatus] = useState('ALL')
  const [search, setSearch] = useState('')
  const [selectedAlert, setSelectedAlert] = useState(null)
  const [page, setPage] = useState(1)
  const PAGE_SIZE = 50

  const filtered = alerts.filter(a => {
    if (filterPattern !== 'ALL' && a.detected_pattern !== filterPattern) return false
    if (filterSeverity !== 'ALL' && a.severity !== filterSeverity) return false
    if (filterStatus !== 'ALL' && a.status !== filterStatus) return false
    if (search) {
      const q = search.toLowerCase()
      if (!((a.from_account || '').toLowerCase().includes(q) ||
            (a.to_account || '').toLowerCase().includes(q) ||
            (a.detected_pattern || '').toLowerCase().includes(q) ||
            (a.tx_id || '').toLowerCase().includes(q))) return false
    }
    return true
  })

  const paginated = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)
  const totalPages = Math.ceil(filtered.length / PAGE_SIZE)

  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-title">Alert Management</div>
          <div className="page-subtitle">{filtered.length} alerts • Click any row to review</div>
        </div>
      </div>

      <div className="page-body">
        {/* Filters */}
        <div className="card" style={{ padding: '14px 20px' }}>
          <div className="filter-bar">
            <input
              className="search-input"
              placeholder="🔍 Search account, pattern, tx..."
              value={search}
              onChange={e => setSearch(e.target.value)}
            />

            <span className="text-muted" style={{ fontSize: 11 }}>Pattern:</span>
            {PATTERNS.slice(0, 5).map(p => (
              <button key={p} className={`filter-chip${filterPattern === p ? ' active' : ''}`}
                onClick={() => setFilterPattern(p)}>{p}</button>
            ))}

            <span className="text-muted" style={{ fontSize: 11 }}>Severity:</span>
            {SEVERITIES.map(s => (
              <button key={s} className={`filter-chip${filterSeverity === s ? ' active' : ''}`}
                onClick={() => setFilterSeverity(s)}>{s}</button>
            ))}

            <span className="text-muted" style={{ fontSize: 11 }}>Status:</span>
            {STATUSES.map(s => (
              <button key={s} className={`filter-chip${filterStatus === s ? ' active' : ''}`}
                onClick={() => setFilterStatus(s)}>{s}</button>
            ))}
          </div>
        </div>

        {/* Table */}
        <div className="card" style={{ padding: 0 }}>
          <div className="alerts-table-container">
            <table className="alerts-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Pattern</th>
                  <th>From Account</th>
                  <th>To Account</th>
                  <th>Amount (USD)</th>
                  <th>Score</th>
                  <th>Severity</th>
                  <th>Type</th>
                  <th>Status</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {paginated.map(alert => (
                  <tr key={alert.id} onClick={() => setSelectedAlert(alert)}
                      style={{ cursor: 'pointer' }}>
                    <td className="mono" style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                      {alert.timestamp ? new Date(alert.timestamp).toLocaleString() : '--'}
                    </td>
                    <td>
                      <span className={`pattern-badge ${alert.detected_pattern}`}>
                        {alert.detected_pattern || 'UNKNOWN'}
                      </span>
                    </td>
                    <td className="mono" style={{ fontSize: 11 }}>
                      <span className="account-chip">{(alert.from_account || '').slice(0, 12)}</span>
                    </td>
                    <td className="mono" style={{ fontSize: 11 }}>
                      <span className="account-chip">{(alert.to_account || '').slice(0, 12)}</span>
                    </td>
                    <td className="mono" style={{ fontSize: 12 }}>
                      ${(alert.amount_usd || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </td>
                    <td>
                      <div className="score-bar-wrap">
                        <div className="score-bar" style={{ width: 60 }}>
                          <div className="score-fill" style={{
                            width: `${Math.round((alert.final_score || 0) * 100)}%`,
                            background: (alert.final_score || 0) > 0.7 ? 'var(--red)' :
                              (alert.final_score || 0) > 0.5 ? 'var(--yellow)' : 'var(--green)'
                          }} />
                        </div>
                        <span className="score-text">{Math.round((alert.final_score || 0) * 100)}%</span>
                      </div>
                    </td>
                    <td><span className={`severity-badge ${alert.severity}`}>{alert.severity}</span></td>
                    <td>
                      {alert.retroactive
                        ? <span className="retro-tag">↩ RETRO</span>
                        : <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>LIVE</span>
                      }
                    </td>
                    <td><span className={`status-badge ${alert.status}`}>{alert.status}</span></td>
                    <td onClick={e => e.stopPropagation()}>
                      {alert.status === 'PENDING' && (
                        <div className="flex-center gap-2">
                          <button className="btn btn-success btn-sm"
                            onClick={() => reviewAlert(alert.id, 'APPROVED', '', alert.involved_accounts || [])}>
                            ✓ Approve
                          </button>
                          <button className="btn btn-danger btn-sm"
                            onClick={() => reviewAlert(alert.id, 'REJECTED')}>
                            ✗ Reject
                          </button>
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {paginated.length === 0 && (
              <div style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-muted)' }}>
                <div style={{ fontSize: 40 }}>🔍</div>
                <div style={{ marginTop: 12, fontSize: 14 }}>No alerts match your filters.</div>
              </div>
            )}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div style={{ padding: '12px 20px', borderTop: '1px solid var(--border)',
                         display: 'flex', alignItems: 'center', gap: 8, justifyContent: 'center' }}>
              <button className="btn btn-ghost btn-sm" onClick={() => setPage(p => Math.max(1, p - 1))}
                disabled={page === 1}>← Prev</button>
              <span className="text-muted" style={{ fontSize: 12 }}>
                Page {page} of {totalPages}
              </span>
              <button className="btn btn-ghost btn-sm" onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                disabled={page === totalPages}>Next →</button>
            </div>
          )}
        </div>
      </div>

      {selectedAlert && (
        <AlertModal
          alert={selectedAlert}
          onClose={() => setSelectedAlert(null)}
          onReview={async (action, note, accounts) => {
            await reviewAlert(selectedAlert.id, action, note, accounts)
            setSelectedAlert(null)
          }}
        />
      )}
    </div>
  )
}
