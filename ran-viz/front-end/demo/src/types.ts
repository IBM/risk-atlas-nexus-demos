//import { strict } from "assert";

export interface NodeData {
  key: string;
  label: string;
  tag: string;
  URL: string;
  cluster: string;
  x: number;
  y: number;
}

export interface EdgeData {
  id: string;
  source: string;
  target: string;
  edge_type: string;
  label: string;
}

export interface Cluster {
  key: string;
  color: string;
  clusterLabel: string;
}

export interface Tag {
  key: string;
  image: string;
}

export interface Dataset {
  nodes: NodeData[];
  edges: EdgeData[];
  clusters: Cluster[];
  tags: Tag[];
}

export interface FiltersState {
  clusters: Record<string, boolean>;
  tags: Record<string, boolean>;
}
