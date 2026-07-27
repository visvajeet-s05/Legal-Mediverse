require("@nomicfoundation/hardhat-toolbox");
require("hardhat-gas-reporter");
require("hardhat-contract-sizer");
require("dotenv").config({ path: process.env.ENV_FILE || ".env.testnet" });

const PRIVATE_KEY = process.env.WEB3_PRIVATE_KEY || process.env.DEPLOYER_PRIVATE_KEY || "0x0000000000000000000000000000000000000000000000000000000000000000";
const AMOY_RPC_URL = process.env.POLYGON_AMOY_RPC_URL || "https://rpc-amoy.polygon.technology";
const MAINNET_RPC_URL = process.env.POLYGON_MAINNET_RPC_URL || "https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY";

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.20",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200,
      },
    },
  },
  paths: {
    sources: "./contracts",
    tests: "./contracts/test",
    cache: "./contracts/cache",
    artifacts: "./contracts/artifacts"
  },
  gasReporter: {
    enabled: process.env.REPORT_GAS === "true",
    currency: "USD",
    token: "MATIC",
    coinmarketcap: process.env.COINMARKETCAP_API_KEY,
    gasPrice: 30,
    showTimeSpent: true,
  },
  contractSizer: {
    alphaSort: true,
    disambiguatePaths: false,
    runOnCompile: true,
    strict: true,
  },
  networks: {
    hardhat: {
      chainId: 1337,
    },
    amoy: {
      url: AMOY_RPC_URL,
      accounts: /^0x[0-9a-fA-F]{64}$/.test(PRIVATE_KEY) ? [PRIVATE_KEY] : [],
      chainId: 80002,
    },
    polygonAmoy: {
      url: AMOY_RPC_URL,
      accounts: /^0x[0-9a-fA-F]{64}$/.test(PRIVATE_KEY) ? [PRIVATE_KEY] : [],
      chainId: 80002,
    },
    polygon: {
      url: MAINNET_RPC_URL,
      accounts: /^0x[0-9a-fA-F]{64}$/.test(PRIVATE_KEY) ? [PRIVATE_KEY] : [],
      chainId: 137,
    },
    polygonMainnet: {
      url: MAINNET_RPC_URL,
      accounts: /^0x[0-9a-fA-F]{64}$/.test(PRIVATE_KEY) ? [PRIVATE_KEY] : [],
      chainId: 137,
    },
  },
  etherscan: {
    apiKey: {
      polygonAmoy: process.env.POLYGONSCAN_API_KEY || "",
      polygon: process.env.POLYGONSCAN_API_KEY || "",
    },
    customChains: [
      {
        network: "polygonAmoy",
        chainId: 80002,
        urls: {
          apiURL: "https://api-amoy.polygonscan.com/api",
          browserURL: "https://amoy.polygonscan.com",
        },
      },
      {
        network: "polygon",
        chainId: 137,
        urls: {
          apiURL: "https://api.polygonscan.com/api",
          browserURL: "https://polygonscan.com",
        },
      },
    ],
  },
};
