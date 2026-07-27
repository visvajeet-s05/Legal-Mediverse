const { ethers, network } = require("hardhat");

/**
 * ⏱️ TimelockController Deployment Script
 * ==========================================
 * Deploys an OpenZeppelin TimelockController between the target contract
 * and Gnosis Safe multi-sig on Polygon Mainnet.
 *
 * Ownership Chain:
 *   Gnosis Safe (Multi-Sig) → TimelockController (48h delay) → Target Contract
 *
 * Usage:
 *   npx hardhat run scripts/deploy-timelock.js --network polygon
 *
 * Environment Variables (.env.production):
 *   GNOSIS_SAFE_ADDRESS - The deployed Gnosis Safe multi-sig address
 *   TIMELOCK_MIN_DELAY  - Delay in seconds (default: 172800 = 2 days)
 */

const DEFAULT_MIN_DELAY = 2 * 24 * 3600; // 2 days in seconds
const ZERO_ADDRESS = "0x0000000000000000000000000000000000000000";

async function main() {
  console.log(`\n==================================================`);
  console.log(`⏱️  Deploying TimelockController on ${network.name}`);
  console.log(`Chain ID: ${network.config.chainId}`);
  console.log(`==================================================\n`);

  // ── 1. Validate Configuration ──────────────────────────────────────
  const gnosisSafeAddress = process.env.GNOSIS_SAFE_ADDRESS;
  if (!gnosisSafeAddress || !ethers.isAddress(gnosisSafeAddress)) {
    throw new Error(
      "❌ Missing or invalid GNOSIS_SAFE_ADDRESS in .env.production.\n" +
      "   Set it to your deployed Gnosis Safe multi-sig address."
    );
  }

  const safeAddress = ethers.getAddress(gnosisSafeAddress);
  const minDelay = parseInt(process.env.TIMELOCK_MIN_DELAY || String(DEFAULT_MIN_DELAY), 10);

  console.log(`🛡️  Gnosis Safe Address:  ${safeAddress}`);
  console.log(`⏱️  Timelock Min Delay:    ${minDelay}s (${minDelay / 86400} days)`);

  const [deployer] = await ethers.getSigners();
  console.log(`👤  Deployer:              ${deployer.address}`);

  // ── 2. Deploy TimelockController ───────────────────────────────────
  console.log("\n🚀 Deploying TimelockController...");

  const Timelock = await ethers.getContractFactory("TimelockController");

  // Configuration:
  // - proposers: [Gnosis Safe] can propose and cancel operations
  // - executors: [Gnosis Safe] can execute ready proposals
  // - admin: Gnosis Safe (temporary - will be renounced later)
  const proposers = [safeAddress];
  const executors = [safeAddress];
  const admin = safeAddress;

  const timelock = await Timelock.deploy(minDelay, proposers, executors, admin);
  await timelock.waitForDeployment();

  const timelockAddress = await timelock.getAddress();
  const deploymentTx = timelock.deploymentTransaction();
  const receipt = await deploymentTx.wait();

  console.log(`\n✅ TimelockController Deployed!`);
  console.log(`   Contract Address: ${timelockAddress}`);
  console.log(`   Deploy Tx:        ${deploymentTx.hash}`);
  console.log(`   Block Number:     ${receipt.blockNumber}`);

  // ── 3. Verify Roles Assigned Correctly ─────────────────────────────
  const PROPOSER_ROLE = await timelock.PROPOSER_ROLE();
  const EXECUTOR_ROLE = await timelock.EXECUTOR_ROLE();
  const CANCELLER_ROLE = await timelock.CANCELLER_ROLE();
  const DEFAULT_ADMIN_ROLE = "0x0000000000000000000000000000000000000000000000000000000000000000";

  const isProposer = await timelock.hasRole(PROPOSER_ROLE, safeAddress);
  const isExecutor = await timelock.hasRole(EXECUTOR_ROLE, safeAddress);
  const isCanceller = await timelock.hasRole(CANCELLER_ROLE, safeAddress);
  const isAdmin = await timelock.hasRole(DEFAULT_ADMIN_ROLE, safeAddress);

  console.log(`\n🔐 Role Verification:`);
  console.log(`   PROPOSER_ROLE:    ${isProposer ? "✅" : "❌"}`);
  console.log(`   EXECUTOR_ROLE:    ${isExecutor ? "✅" : "❌"}`);
  console.log(`   CANCELLER_ROLE:   ${isCanceller ? "✅" : "❌"}`);
  console.log(`   DEFAULT_ADMIN:    ${isAdmin ? "⚠️  (should be renounced)" : "✅ (already none)"}`);

  // ── 4. Post-Deployment Summary ──────────────────────────────────────
  console.log(`\n==================================================`);
  console.log(`📋  NEXT STEPS`);
  console.log(`==================================================`);
  console.log(`\n1. Add to .env.production:`);
  console.log(`   TIMELOCK_CONTRACT_ADDRESS=${timelockAddress}`);
  console.log(`\n2. Transfer contract ownership to Timelock:`);
  console.log(`   npx hardhat run scripts/transfer-to-timelock.js --network ${network.name}`);
  console.log(`\n3. Renounce DEFAULT_ADMIN_ROLE on Timelock:`);
  console.log(`   - Open Gnosis Safe (${safeAddress})`);
  console.log(`   - Use Transaction Builder to call timelock.renounceRole(DEFAULT_ADMIN_ROLE, safeAddress)`);
  console.log(`\n4. Set TIMELOCK_CONTRACT_ADDRESS in frontend .env`);
  console.log(`==================================================\n`);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("❌ Deployment failed:", error);
    process.exit(1);
  });
