import { http, createConfig } from "wagmi";
import { mainnet, polygon, polygonAmoy } from "wagmi/chains";
import { injected } from "wagmi/connectors";

const amoyRpc = process.env.NEXT_PUBLIC_POLYGON_AMOY_RPC || process.env.NEXT_PUBLIC_RPC_URL || "https://rpc-amoy.polygon.technology";

export const config = createConfig({
  chains: [mainnet, polygon, polygonAmoy],
  connectors: [
    injected(),
  ],
  transports: {
    [mainnet.id]: http(),
    [polygon.id]: http(),
    [polygonAmoy.id]: http(amoyRpc),
  },
});

export { MEDICAL_ESCROW_ABI, MEDICAL_ESCROW_CONTRACT_ADDRESS } from "./contracts";
