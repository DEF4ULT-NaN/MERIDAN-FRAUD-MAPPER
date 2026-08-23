// Reproduce the exact adapter logic against the real /sample-dataset payload
// to see if it returns a valid {nodes, edges, analysis} object.
const http = require('http');

function getJson(path) {
  return new Promise((resolve, reject) => {
    http.get('http://127.0.0.1:8000' + path, (res) => {
      let data = '';
      res.on('data', (c) => data += c);
      res.on('end', () => {
        try { resolve(JSON.parse(data)); }
        catch (e) { reject(new Error('JSON parse: ' + e.message + ' body=' + data.slice(0,200))); }
      });
    }).on('error', reject);
  });
}

function adaptBackendPayloadToGraph(p) {
  console.log('[node] adapt: p.nodes =', p && p.nodes && p.nodes.length, 'p.edges =', p && p.edges && p.edges.length);
  const idToNode = new Map();
  const nodes = p.nodes.map(n => {
    const risk = Math.round((n.risk_score || 0) * 100);
    const degree = (n.features && n.features.degree) || 0;
    const node = {
      id: n.id, name: n.name || n.id,
      risk, riskBand: risk >= 70 ? 'high' : risk >= 40 ? 'mid' : 'low',
      degree, distinctAttrTypes: 0,
      accountAgeDays: (n.features && n.features.account_age_days) || 0,
      avgTxGapMin: 0,
      ring: n.ring_id || -1, ringId: n.ring_id || null,
      seedFraud: !!n.is_fraud,
    };
    idToNode.set(n.id, node);
    return node;
  });
  const edges = p.edges.map(e => ({
    source: e.source, target: e.target,
    shared: e.attribute_types || [],
    weight: e.weight || 1, reasons: e.reasons || [],
  }));
  const counts = new Map();
  edges.forEach(e => {
    const t = e.shared.length;
    counts.set(e.source, Math.max(counts.get(e.source) || 0, t));
    counts.set(e.target, Math.max(counts.get(e.target) || 0, t));
  });
  nodes.forEach(n => n.distinctAttrTypes = counts.get(n.id) || 0);
  const adj = new Map(nodes.map(n => [n.id, []]));
  edges.forEach(e => { adj.get(e.source).push(e); adj.get(e.target).push(e); });
  const visited = new Set();
  const clusters = [];
  nodes.forEach(n => {
    if (visited.has(n.id)) return;
    const stack = [n.id]; const comp = [];
    visited.add(n.id);
    while (stack.length) {
      const cur = stack.pop(); comp.push(cur);
      adj.get(cur).forEach(l => {
        const other = l.source === cur ? l.target : l.source;
        if (!visited.has(other)) { visited.add(other); stack.push(other); }
      });
    }
    clusters.push(comp);
  });
  clusters.forEach((comp, ci) => comp.forEach(id => idToNode.get(id).clusterId = ci));
  const fraudClusters = clusters.filter(c => {
    if (c.length < 3) return false;
    return c.reduce((s, id) => s + idToNode.get(id).risk, 0) / c.length >= 55;
  });
  return { nodes, edges, analysis: { adj, clusters, fraudClusters } };
}

(async () => {
  const sample = await getJson('/sample-dataset');
  console.log('Got sample, top-level keys =', Object.keys(sample));
  console.log('  stats =', sample.stats);
  console.log('  first node =', sample.nodes[0]);
  console.log('  first edge =', sample.edges[0]);

  const result = adaptBackendPayloadToGraph(sample);
  console.log('\nAdapter returned:');
  console.log('  result.nodes length =', result.nodes && result.nodes.length);
  console.log('  result.edges length =', result.edges && result.edges.length);
  console.log('  result.analysis keys =', Object.keys(result.analysis || {}));
  console.log('  result.analysis.fraudClusters length =', result.analysis.fraudClusters.length);

  // Now try what renderGraph would do
  console.log('\nSimulating renderGraph first lines:');
  console.log('  graph.nodes.map exists?', typeof result.nodes.map);
  console.log('  graph.edges.map exists?', typeof result.edges.map);
})().catch(e => {
  console.error('ERROR:', e.message);
  console.error(e.stack);
});