import networkx as nx

def is_d_separated_custom(nodes, edges, u, v, Z):
    """
    Computes exact d-separation using standard Bayes Ball / active path reachability.
    """
    Z = set(Z)
    DG = nx.DiGraph()
    DG.add_nodes_from(nodes)
    for src, tgt in edges:
        DG.add_edge(src, tgt)

    Z_and_descendants = set(Z)
    for z_node in list(Z):
        if z_node in DG:
            Z_and_descendants.update(nx.descendants(DG, z_node))

    visited = set()
    queue = [(u, 'from_child')]
    
    while queue:
        curr, dir_type = queue.pop(0)
        state = (curr, dir_type)
        if state in visited:
            continue
        visited.add(state)
        
        if curr == v:
            return False  # Open path found -> NOT d-separated
            
        if dir_type == 'from_child':
            if curr not in Z:
                for p in DG.predecessors(curr):
                    queue.append((p, 'from_child'))
                for c in DG.successors(curr):
                    queue.append((c, 'from_parent'))
        elif dir_type == 'from_parent':
            if curr not in Z:
                for c in DG.successors(curr):
                    queue.append((c, 'from_parent'))
            if curr in Z_and_descendants:
                for p in DG.predecessors(curr):
                    queue.append((p, 'from_child'))
                    
    return True
