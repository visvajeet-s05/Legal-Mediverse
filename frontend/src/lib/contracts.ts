// Deployed Contract Address configuration
export const MEDICAL_ESCROW_CONTRACT_ADDRESS =
  process.env.NEXT_PUBLIC_ESCROW_CONTRACT_ADDRESS || "0x5FbDB2315678afecb367f032d93F642f64180aa3"; // Local Hardhat Default

// Full ABI — auto-generated from contracts/MedicalEscrow.sol v2 (M2)
// Includes all AccessControl inherited functions, role constants, core escrow
// functions, and events for complete frontend ↔ contract wiring.
export const MEDICAL_ESCROW_ABI = [
  // ─── Constructor ────────────────────────────────────────────────────────
  {
    "inputs": [
      { "internalType": "address", "name": "initialOracle", "type": "address" },
      { "internalType": "bytes32", "name": "_jobId", "type": "bytes32" },
      { "internalType": "uint256", "name": "_fee", "type": "uint256" }
    ],
    "stateMutability": "nonpayable",
    "type": "constructor"
  },

  // ─── Role Constants (bytes32) ──────────────────────────────────────────
  {
    "inputs": [],
    "name": "ORACLE_ROLE",
    "outputs": [{ "internalType": "bytes32", "name": "", "type": "bytes32" }],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "RELEASE_ROLE",
    "outputs": [{ "internalType": "bytes32", "name": "", "type": "bytes32" }],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "PAUSER_ROLE",
    "outputs": [{ "internalType": "bytes32", "name": "", "type": "bytes32" }],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "DEFAULT_ADMIN_ROLE",
    "outputs": [{ "internalType": "bytes32", "name": "", "type": "bytes32" }],
    "stateMutability": "view",
    "type": "function"
  },

  // ─── AccessControl (inherited) ─────────────────────────────────────────
  {
    "inputs": [],
    "name": "getRoleAdmin",
    "outputs": [{ "internalType": "bytes32", "name": "", "type": "bytes32" }],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [
      { "internalType": "bytes32", "name": "role", "type": "bytes32" },
      { "internalType": "address", "name": "account", "type": "address" }
    ],
    "name": "grantRole",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      { "internalType": "bytes32", "name": "role", "type": "bytes32" },
      { "internalType": "address", "name": "account", "type": "address" }
    ],
    "name": "hasRole",
    "outputs": [{ "internalType": "bool", "name": "", "type": "bool" }],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [
      { "internalType": "bytes32", "name": "role", "type": "bytes32" },
      { "internalType": "address", "name": "account", "type": "address" }
    ],
    "name": "renounceRole",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      { "internalType": "bytes32", "name": "role", "type": "bytes32" },
      { "internalType": "address", "name": "account", "type": "address" }
    ],
    "name": "revokeRole",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      { "internalType": "bytes32", "name": "role", "type": "bytes32" },
      { "internalType": "uint256", "name": "index", "type": "uint256" }
    ],
    "name": "getRoleMember",
    "outputs": [{ "internalType": "address", "name": "", "type": "address" }],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [{ "internalType": "bytes32", "name": "role", "type": "bytes32" }],
    "name": "getRoleMemberCount",
    "outputs": [{ "internalType": "uint256", "name": "", "type": "uint256" }],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [{ "internalType": "bytes4", "name": "interfaceId", "type": "bytes4" }],
    "name": "supportsInterface",
    "outputs": [{ "internalType": "bool", "name": "", "type": "bool" }],
    "stateMutability": "view",
    "type": "function"
  },

  // ─── Pausable ───────────────────────────────────────────────────────────
  {
    "inputs": [],
    "name": "paused",
    "outputs": [{ "internalType": "bool", "name": "", "type": "bool" }],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "pause",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "unpause",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },

  // ─── Contract State ─────────────────────────────────────────────────────
  {
    "inputs": [],
    "name": "oracle",
    "outputs": [{ "internalType": "address", "name": "", "type": "address" }],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "jobId",
    "outputs": [{ "internalType": "bytes32", "name": "", "type": "bytes32" }],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "fee",
    "outputs": [{ "internalType": "uint256", "name": "", "type": "uint256" }],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "campaignCount",
    "outputs": [{ "internalType": "uint256", "name": "", "type": "uint256" }],
    "stateMutability": "view",
    "type": "function"
  },

  // ─── Campaign Struct ────────────────────────────────────────────────────
  {
    "inputs": [{ "internalType": "uint256", "name": "", "type": "uint256" }],
    "name": "campaigns",
    "outputs": [
      { "internalType": "address payable", "name": "creator", "type": "address" },
      { "internalType": "address payable", "name": "hospitalWallet", "type": "address" },
      { "internalType": "uint256", "name": "targetAmount", "type": "uint256" },
      { "internalType": "uint256", "name": "amountRaised", "type": "uint256" },
      { "internalType": "uint256", "name": "billTotalExtracted", "type": "uint256" },
      { "internalType": "uint256", "name": "fraudRiskScore", "type": "uint256" },
      { "internalType": "uint8", "name": "verificationStatus", "type": "uint8" },
      { "internalType": "bool", "name": "isReleased", "type": "bool" }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [
      { "internalType": "uint256", "name": "", "type": "uint256" },
      { "internalType": "address", "name": "", "type": "address" }
    ],
    "name": "contributions",
    "outputs": [{ "internalType": "uint256", "name": "", "type": "uint256" }],
    "stateMutability": "view",
    "type": "function"
  },

  // ─── Core Functions ─────────────────────────────────────────────────────
  {
    "inputs": [
      { "internalType": "address payable", "name": "hospitalWallet", "type": "address" },
      { "internalType": "uint256", "name": "targetAmount", "type": "uint256" }
    ],
    "name": "createCampaign",
    "outputs": [{ "internalType": "uint256", "name": "campaignId", "type": "uint256" }],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [{ "internalType": "uint256", "name": "campaignId", "type": "uint256" }],
    "name": "donate",
    "outputs": [],
    "stateMutability": "payable",
    "type": "function"
  },
  {
    "inputs": [
      { "internalType": "uint256", "name": "campaignId", "type": "uint256" },
      { "internalType": "string", "name": "billReference", "type": "string" }
    ],
    "name": "requestBillVerification",
    "outputs": [{ "internalType": "bytes32", "name": "requestId", "type": "bytes32" }],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      { "internalType": "uint256", "name": "campaignId", "type": "uint256" },
      { "internalType": "bool", "name": "isVerified", "type": "bool" },
      { "internalType": "uint256", "name": "billTotalExtracted", "type": "uint256" },
      { "internalType": "uint256", "name": "fraudRiskScore", "type": "uint256" }
    ],
    "name": "fulfillBillVerification",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [{ "internalType": "uint256", "name": "campaignId", "type": "uint256" }],
    "name": "releaseFunds",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [{ "internalType": "uint256", "name": "campaignId", "type": "uint256" }],
    "name": "claimRefund",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },

  // ─── Events ─────────────────────────────────────────────────────────────
  {
    "anonymous": false,
    "inputs": [
      { "indexed": true, "internalType": "uint256", "name": "campaignId", "type": "uint256" },
      { "indexed": true, "internalType": "address", "name": "creator", "type": "address" },
      { "indexed": true, "internalType": "address", "name": "hospital", "type": "address" },
      { "indexed": false, "internalType": "uint256", "name": "targetAmount", "type": "uint256" }
    ],
    "name": "CampaignCreated",
    "type": "event"
  },
  {
    "anonymous": false,
    "inputs": [
      { "indexed": true, "internalType": "uint256", "name": "campaignId", "type": "uint256" },
      { "indexed": true, "internalType": "address", "name": "donor", "type": "address" },
      { "indexed": false, "internalType": "uint256", "name": "amount", "type": "uint256" }
    ],
    "name": "DonationReceived",
    "type": "event"
  },
  {
    "anonymous": false,
    "inputs": [
      { "indexed": true, "internalType": "uint256", "name": "campaignId", "type": "uint256" },
      { "indexed": true, "internalType": "bytes32", "name": "requestId", "type": "bytes32" }
    ],
    "name": "BillVerificationRequested",
    "type": "event"
  },
  {
    "anonymous": false,
    "inputs": [
      { "indexed": true, "internalType": "uint256", "name": "campaignId", "type": "uint256" },
      { "indexed": false, "internalType": "bool", "name": "isVerified", "type": "bool" },
      { "indexed": false, "internalType": "uint256", "name": "billTotal", "type": "uint256" },
      { "indexed": false, "internalType": "uint256", "name": "fraudRiskScore", "type": "uint256" }
    ],
    "name": "BillVerificationFulfilled",
    "type": "event"
  },
  {
    "anonymous": false,
    "inputs": [
      { "indexed": true, "internalType": "uint256", "name": "campaignId", "type": "uint256" },
      { "indexed": true, "internalType": "address", "name": "hospital", "type": "address" },
      { "indexed": false, "internalType": "uint256", "name": "amount", "type": "uint256" }
    ],
    "name": "FundsReleased",
    "type": "event"
  },
  {
    "anonymous": false,
    "inputs": [
      { "indexed": true, "internalType": "uint256", "name": "campaignId", "type": "uint256" },
      { "indexed": true, "internalType": "address", "name": "donor", "type": "address" },
      { "indexed": false, "internalType": "uint256", "name": "amount", "type": "uint256" }
    ],
    "name": "RefundIssued",
    "type": "event"
  }
] as const;

// ─── Verification Status Enum Mapping ─────────────────────────────────────
export const VERIFICATION_STATUS = {
  Pending: 0,
  Approved: 1,
  Rejected: 2,
} as const;

export type VerificationStatus = keyof typeof VERIFICATION_STATUS;
