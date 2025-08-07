import { FC } from "react";
import { BsInfoCircle } from "react-icons/bs";

import Panel from "./Panel";

const DescriptionPanel: FC = () => {
  return (
    <Panel
      initiallyDeployed
      title={
        <>
          <BsInfoCircle className="text-muted" /> Description
        </>
      }
    >
      <p>
        This map represents a <i>network</i> of the nodes in the Risk Atlas Nexus graph. Each{" "}
        <i>node</i> represents a graph entity, and each edge a particular type of relationship between them
        .
      </p>
      <p>
        Nodes sizes are related to their{" "}
        <a target="_blank" rel="noreferrer" href="https://en.wikipedia.org/wiki/Betweenness_centrality">
          betweenness centrality
        </a>
        . More central nodes (ie. bigger nodes) are important crossing points in the network. Finally, You can click a
        node to open the related uri.
      </p>
      <p>
        This visualisation implemation was based on a sigma js demo {" "}
        <a target="_blank" rel="noreferrer" href="https://github.com/jacomyal/sigma.js/tree/main/packages/demo">
          on GitHub
        </a>
        .
      </p>
    </Panel>
  );
};

export default DescriptionPanel;
