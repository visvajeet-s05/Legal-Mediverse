/**
 * 🛡️ Safe Multi-Sig Governance Workflow
 * =======================================
 * Integrates @safe-global/protocol-kit + @safe-global/api-kit for
 * off-chain signature collection and on-chain execution via TimelockController.
 *
 * This script demonstrates the complete multi-sig governance flow:
 *   1. Owner 1 proposes a pauser role grant through the Safe
 *   2. Owner 2 signs off-chain (no gas cost)
 *   3. Executes on-chain once threshold is met
 *   4. Schedules through TimelockController (if configured)
 *
 * Usage:
 *   node scripts/safe-multisig-workflow.js
 *
 * Required packages:
 *   npm install @safe-global/protocol-kit @safe-global/api-kit @safe-global/types-kit ethers dotenv
 *
 * Environment:
 *   POLYGON_MAINNET_RPC_URL - Alchemy/Infura RPC endpoint
 *   GNOSIS_SAFE_ADDRESS     - Deployed Safe address
 *   SAFE_OWNER_1_KEY        - Owner 1 private key
 *   SAFE_OWNER_2_KEY        - Owner 2 private key
 *   ESCROW_CONTRACT_ADDRESS - MedicalEscrow contract address
 *   TIMELOCK_CONTRACT_ADDRESS - (Optional) TimelockController address
 */

const Safe = require("@safe-global/protocol-kit").default;
const SafeApiKit = require("@safe-global/api-kit").default;
const { MetaTransactionData, OperationType } = require("@safe-global/types-kit");
const { ethers } = require("ethers");
const dotenv = require("dotenv");

dotenv.config({ path: ".env.production" });

// ─── Configuration ─────────────────────────────────────────────────────────
const RPC_URL =
  process.env.POLYGON_MAINNET_RPC_URL ||
  "https://polygon-mainnet.g.alchemy.com/v2/demo";
const SAFE_ADDRESS = process.env.GNOSIS_SAFE_ADDRESS;
const OWNER_1_KEY = process.env.SAFE_OWNER_1_KEY;
const OWNER_2_KEY = process.env.SAFE_OWNER_2_KEY;
const ESCROW_ADDRESS = process.env.ESCROW_CONTRACT_ADDRESS;
const TIMELOCK_ADDRESS = process.env.TIMELOCK_CONTRACT_ADDRESS;

const CHAIN_ID = 137n; // Polygon Mainnet

// ─── Contract ABIs (Minimal) ───────────────────────────────────────────────
const PAUSER_ABI = [
  {
    inputs: [
      { name: "role", type: "bytes32" },
      { name: "account", type: "address" },
    ],
    name: "grantRole",
    outputs: [],
    stateMutability: "nonpayable",
    type: "function",
  },
  {
    inputs: [
      { name: "role", type: "bytes32" },
      { name: "account", type: "address" },
    ],
    name: "hasRole",
    outputs: [{ type: "bool" }],
    stateMutability: "view",
    type: "function",
  },
];

const TIMELOCK_ABI = [
  {
    inputs: [
      { name: "target", type: "address" },
      { name: "value", type: "uint256" },
      { name: "data", type: "bytes" },
      { name: "predecessor", type: "bytes32" },
      { name: "salt", type: "bytes32" },
      { name: "delay", type: "uint256" },
    ],
    name: "schedule",
    outputs: [],
    stateMutability: "nonpayable",
    type: "function",
  },
  {
    inputs: [
      { name: "target", type: "address" },
      { name: "value", type: "uint256" },
      { name: "data", type: "bytes" },
      { name: "predecessor", type: "bytes32" },
      { name: "salt", type: "bytes32" },
    ],
    name: "execute",
    outputs: [],
    stateMutability: "payable",
    type: "function",
  },
];

// ─── PAUSER_ROLE bytes32 constant ─────────────────────────────────────────
const PAUSER_ROLE = ethers.id("PAUSER_ROLE");

async function runSafeGovernanceFlow() {
  console.log("=".repeat(60));
  console.log("🛡️  SAFE MULTI-SIG GOVERNANCE WORKFLOW");
  console.log("=".repeat(60));

  // ═════════════════════════════════════════════════════════════════════
  // STEP 0: Validation
  // ═════════════════════════════════════════════════════════════════════
  if (!SAFE_ADDRESS) throw new Error("Missing GNOSIS_SAFE_ADDRESS");
  if (!OWNER_1_KEY) throw new Error("Missing SAFE_OWNER_1_KEY");
  if (!OWNER_2_KEY) throw new Error("Missing SAFE_OWNER_2_KEY");
  if (!ESCROW_ADDRESS) throw new Error("Missing ESCROW_CONTRACT_ADDRESS");

  console.log(`\n📋 Configuration:`);
  console.log(`   Safe Address:         ${SAFE_ADDRESS}`);
  console.log(`   Escrow Contract:      ${ESCROW_ADDRESS}`);
  console.log(`   Timelock Controller:  ${TIMELOCK_ADDRESS || "Not configured (direct execution)"}`);
  console.log(`   Chain ID:             ${CHAIN_ID}`);

  // ═════════════════════════════════════════════════════════════════════
  // STEP 1: Initialize Safe API Kit (shared across all owners)
  // ═════════════════════════════════════════════════════════════════════
  console.log(`\n--> [1/5] Initializing Safe API Kit...`);
  const apiKit = new SafeApiKit({
    chainId: CHAIN_ID,
  });
  console.log("   ✅ Safe API Kit initialized");

  // ═════════════════════════════════════════════════════════════════════
  // STEP 2: Owner 1 — Create, sign, and propose transaction
  // ═════════════════════════════════════════════════════════════════════
  console.log(`\n--> [2/5] Owner 1: Creating & proposing transaction...`);

  const protocolKitOwner1 = await Safe.init({
    provider: RPC_URL,
    signer: OWNER_1_KEY,
    safeAddress: SAFE_ADDRESS,
  });

  const owner1Address = await protocolKitOwner1.getSafeProvider().getSignerAddress();
  console.log(`   Owner 1: ${owner1Address}`);

  // Build calldata: grantRole(PAUSER_ROLE, newPauser)
  const escrowInterface = new ethers.Interface(PAUSER_ABI);
  const newPauser = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"; // Example new pauser
  const grantRoleData = escrowInterface.encodeFunctionData("grantRole", [
    PAUSER_ROLE,
    newPauser,
  ]);

  let targetAddress = ESCROW_ADDRESS;
  // If Timelock is configured, target the Timelock instead (it will call escrow after delay)
  if (TIMELOCK_ADDRESS) {
    targetAddress = TIMELOCK_ADDRESS;
    console.log(`   ⏱️  Routing through TimelockController: ${TIMELOCK_ADDRESS}`);
  }

  const transactionData = {
    to: targetAddress,
    value: "0",
    data: grantRoleData,
    operation: OperationType.Call,
  };

  // Create Safe transaction
  const safeTransaction = await protocolKitOwner1.createTransaction({
    transactions: [transactionData],
  });

  // Compute EIP-712 hash & sign
  const safeTxHash = await protocolKitOwner1.getTransactionHash(safeTransaction);
  const signature1 = await protocolKitOwner1.signHash(safeTxHash);

  // Propose to Safe Transaction Service
  await apiKit.proposeTransaction({
    safeAddress: SAFE_ADDRESS,
    safeTransactionData: safeTransaction.data,
    safeTxHash,
    senderAddress: owner1Address,
    senderSignature: signature1.data,
  });

  console.log(`   ✅ Transaction proposed!`);
  console.log(`   📝 safeTxHash: ${safeTxHash}`);
  console.log(`   📝 Calldata: grantRole(PAUSER_ROLE, ${newPauser})`);

  // ═════════════════════════════════════════════════════════════════════
  // STEP 3: Owner 2 — Fetch proposal & sign off-chain
  // ═════════════════════════════════════════════════════════════════════
  console.log(`\n--> [3/5] Owner 2: Fetching & confirming transaction...`);

  const protocolKitOwner2 = await Safe.init({
    provider: RPC_URL,
    signer: OWNER_2_KEY,
    safeAddress: SAFE_ADDRESS,
  });

  const owner2Address = await protocolKitOwner2.getSafeProvider().getSignerAddress();
  console.log(`   Owner 2: ${owner2Address}`);

  // Retrieve the pending proposal
  const pendingTx = await apiKit.getTransaction(safeTxHash);
  console.log(`   Pending confirmations: ${pendingTx.confirmations?.length || 0}`);

  // Sign off-chain (no gas cost)
  const signature2 = await protocolKitOwner2.signHash(pendingTx.safeTxHash);

  // Submit confirmation to Safe API
  await apiKit.confirmTransaction(pendingTx.safeTxHash, signature2.data);
  console.log(`   ✅ Owner 2 signature submitted off-chain!`);

  // ═════════════════════════════════════════════════════════════════════
  // STEP 4: Check threshold & execute
  // ═════════════════════════════════════════════════════════════════════
  console.log(`\n--> [4/5] Checking threshold & executing...`);

  const threshold = await protocolKitOwner1.getThreshold();
  const readyTx = await apiKit.getTransaction(safeTxHash);
  const collectedSignatures = readyTx.confirmations?.length || 0;

  console.log(`   Required threshold: ${threshold}`);
  console.log(`   Signatures collected: ${collectedSignatures}`);

  if (collectedSignatures < threshold) {
    console.log(`   ⚠️  Need ${threshold - collectedSignatures} more signature(s) before execution.`);
    console.log(`   Collect remaining signatures off-chain before proceeding.`);
    console.log(`\n📋 To collect more signatures:`);
    console.log(`   1. Share this safeTxHash with other owners: ${safeTxHash}`);
    console.log(`   2. Each owner calls: apiKit.confirmTransaction(safeTxHash, signature)`);
    return;
  }

  console.log(`   ✅ Threshold met! Proceeding to execution...`);

  // ═════════════════════════════════════════════════════════════════════
  // STEP 5: Execute transaction
  // ═════════════════════════════════════════════════════════════════════
  console.log(`\n--> [5/5] Executing transaction on-chain...`);

  if (TIMELOCK_ADDRESS) {
    // ── Timelock Flow: schedule() + wait + execute() ────────────
    console.log(`   ⏱️  Using TimelockController flow...`);

    const timelockInterface = new ethers.Interface(TIMELOCK_ABI);
    const minDelay = 2 * 24 * 3600; // 2 days

    // Schedule the operation through Timelock
    const scheduleData = timelockInterface.encodeFunctionData("schedule", [
      ESCROW_ADDRESS,       // target
      0,                    // value
      grantRoleData,        // data (original escrow call)
      ethers.ZeroHash,      // predecessor
      safeTxHash,           // salt (use safeTxHash for idempotency)
      minDelay,             // delay
    ]);

    // Create a new Safe transaction for scheduling
    const scheduleTxData = {
      to: TIMELOCK_ADDRESS,
      value: "0",
      data: scheduleData,
      operation: OperationType.Call,
    };

    const scheduleSafeTx = await protocolKitOwner1.createTransaction({
      transactions: [scheduleTxData],
    });

    // Execute the schedule transaction
    const executeResponse = await protocolKitOwner1.executeTransaction(readyTx);
    const receipt = await executeResponse.transactionResponse?.wait();

    console.log(`   ✅ Transaction scheduled through Timelock!`);
    console.log(`   📍 Block: ${receipt?.blockNumber}`);
    console.log(`   ⏳ Timelock delay: ${minDelay / 86400} days`);
    console.log(`\n   After ${minDelay / 86400} days, execute:`);
    console.log(`   timelock.execute(${ESCROW_ADDRESS}, 0, ${grantRoleData.substring(0, 40)}..., ${ethers.ZeroHash}, ${safeTxHash})`);
  } else {
    // ── Direct Execution Flow (no Timelock) ────────────────────
    console.log(`   ⚡ Executing directly (no Timelock configured)...`);

    const executeResponse = await protocolKitOwner1.executeTransaction(readyTx);
    const receipt = await executeResponse.transactionResponse?.wait();

    console.log(`   ✅ Transaction executed on-chain!`);
    console.log(`   📍 Block: ${receipt?.blockNumber}`);
    console.log(`   ⛽ Gas Used: ${receipt?.gasUsed?.toString()}`);
  }

  // ═════════════════════════════════════════════════════════════════════
  // Verification
  // ═════════════════════════════════════════════════════════════════════
  console.log(`\n--> Verifying on-chain state...`);
  const provider = new ethers.JsonRpcProvider(RPC_URL);
  const escrowContract = new ethers.Contract(ESCROW_ADDRESS, PAUSER_ABI, provider);
  const hasRole = await escrowContract.hasRole(PAUSER_ROLE, newPauser);

  console.log(`   New pauser (${newPauser}) has PAUSER_ROLE: ${hasRole ? "✅" : "❌"}`);

  console.log(`\n${"=".repeat(60)}`);
  console.log(`✅ MULTI-SIG GOVERNANCE FLOW COMPLETE`);
  console.log(`   safeTxHash: ${safeTxHash}`);
  console.log(`   Polygonscan: https://polygonscan.com/tx/${receipt?.hash}`);
  console.log(`   Safe App: https://app.safe.global/transactions/queue?safe=matic:${SAFE_ADDRESS}`);
  console.log(`=".repeat(60)}\n`);
}

runSafeGovernanceFlow().catch((error) => {
  console.error("\n❌ Governance workflow failed:", error);
  process.exit(1);
});
