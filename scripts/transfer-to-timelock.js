const { ethers, network } = require("hardhat");

/**
 * 🔄 Transfer Contract Ownership to TimelockController
 * ======================================================
 * Transfers DEFAULT_ADMIN_ROLE (and other administrative roles) from
 * the deployer to the TimelockController contract.
 *
 * Ownership Chain:
 *   Gnosis Safe (Multi-Sig) → TimelockController (48h delay) → MedicalEscrow
 *
 * Usage:
 *   npx hardhat run scripts/transfer-to-timelock.js --network polygon
 *
 * Environment Variables (.env.production):
 *   ESCROW_CONTRACT_ADDRESS  - Deployed MedicalEscrow contract address
 *   TIMELOCK_CONTRACT_ADDRESS - Deployed TimelockController address
 *   WEB3_PRIVATE_KEY         - Deployer wallet (must have DEFAULT_ADMIN_ROLE)
 */

async function main() {
  console.log(`\n==================================================`);
  console.log(`🔄 Transferring Contract Ownership to Timelock`);
  console.log(`Network: ${network.name} (Chain ID: ${network.config.chainId})`);
  console.log(`==================================================\n`);

  // ── 1. Validate Configuration ──────────────────────────────────────
  const escrowAddress = process.env.ESCROW_CONTRACT_ADDRESS;
  const timelockAddress = process.env.TIMELOCK_CONTRACT_ADDRESS;

  if (!escrowAddress || !ethers.isAddress(escrowAddress)) {
    throw new Error("❌ Missing or invalid ESCROW_CONTRACT_ADDRESS in .env");
  }
  if (!timelockAddress || !ethers.isAddress(timelockAddress)) {
    throw new Error("❌ Missing or invalid TIMELOCK_CONTRACT_ADDRESS in .env.\n" +
      "   Deploy TimelockController first: npx hardhat run scripts/deploy-timelock.js --network polygon");
  }

  const escrowAddr = ethers.getAddress(escrowAddress);
  const timelockAddr = ethers.getAddress(timelockAddress);

  const [deployer] = await ethers.getSigners();
  console.log(`📜 Escrow Contract:     ${escrowAddr}`);
  console.log(`⏱️  Timelock Controller: ${timelockAddr}`);
  console.log(`👤  Deployer:            ${deployer.address}`);

  // ── 2. Attach to MedicalEscrow Contract ─────────────────────────────
  const MedicalEscrow = await ethers.getContractFactory("MedicalEscrow");
  const escrow = MedicalEscrow.attach(escrowAddr);

  // ── 3. Check Current Roles ─────────────────────────────────────────
  const DEFAULT_ADMIN_ROLE = "0x0000000000000000000000000000000000000000000000000000000000000000";
  const PAUSER_ROLE = await escrow.PAUSER_ROLE();
  const RELEASE_ROLE = await escrow.RELEASE_ROLE();
  const ORACLE_ROLE = await escrow.ORACLE_ROLE();

  const hasAdminRole = await escrow.hasRole(DEFAULT_ADMIN_ROLE, deployer.address);
  if (!hasAdminRole) {
    throw new Error("❌ Deployer does NOT have DEFAULT_ADMIN_ROLE on the contract.");
  }

  console.log(`\n🔍 Current Role Status:`);
  console.log(`   Deployer has DEFAULT_ADMIN_ROLE: ✅`);
  console.log(`   Timelock has DEFAULT_ADMIN_ROLE: ${await escrow.hasRole(DEFAULT_ADMIN_ROLE, timelockAddr) ? "✅" : "❌"}`);
  console.log(`   Timelock has PAUSER_ROLE:        ${await escrow.hasRole(PAUSER_ROLE, timelockAddr) ? "✅" : "❌"}`);
  console.log(`   Timelock has RELEASE_ROLE:       ${await escrow.hasRole(RELEASE_ROLE, timelockAddr) ? "✅" : "❌"}`);

  // ── 4. Grant Roles to Timelock Controller ──────────────────────────
  const gasConfig = { maxPriorityFeePerGas: ethers.parseUnits("35", "gwei") };

  // Grant DEFAULT_ADMIN_ROLE (full admin control, subject to timelock delay)
  console.log(`\n🚀 Granting DEFAULT_ADMIN_ROLE to Timelock...`);
  const tx1 = await escrow.grantRole(DEFAULT_ADMIN_ROLE, timelockAddr, gasConfig);
  await tx1.wait(2);
  console.log(`   ✅ Tx: ${tx1.hash}`);

  // Grant PAUSER_ROLE (emergency pause/unpause)
  console.log(`\n🚀 Granting PAUSER_ROLE to Timelock...`);
  const tx2 = await escrow.grantRole(PAUSER_ROLE, timelockAddr, gasConfig);
  await tx2.wait(2);
  console.log(`   ✅ Tx: ${tx2.hash}`);

  // Grant RELEASE_ROLE (funds release)
  console.log(`\n🚀 Granting RELEASE_ROLE to Timelock...`);
  const tx3 = await escrow.grantRole(RELEASE_ROLE, timelockAddr, gasConfig);
  await tx3.wait(2);
  console.log(`   ✅ Tx: ${tx3.hash}`);

  // Grant ORACLE_ROLE if deployer holds it
  const hasOracleRole = await escrow.hasRole(ORACLE_ROLE, deployer.address);
  if (hasOracleRole) {
    console.log(`\n🚀 Granting ORACLE_ROLE to Timelock...`);
    const tx4 = await escrow.grantRole(ORACLE_ROLE, timelockAddr, gasConfig);
    await tx4.wait(2);
    console.log(`   ✅ Tx: ${tx4.hash}`);
  }

  // ── 5. Renounce Admin Role from Deployer ────────────────────────────
  console.log(`\n⚠️  Renouncing DEFAULT_ADMIN_ROLE from deployer...`);
  const tx5 = await escrow.renounceRole(DEFAULT_ADMIN_ROLE, deployer.address, gasConfig);
  await tx5.wait(2);
  console.log(`   ✅ DEFAULT_ADMIN_ROLE renounced from deployer. Tx: ${tx5.hash}`);

  // ── 6. Final Verification ──────────────────────────────────────────
  console.log(`\n📋 Post-Transfer Role Verification:`);
  console.log(`   Timelock has DEFAULT_ADMIN_ROLE: ${await escrow.hasRole(DEFAULT_ADMIN_ROLE, timelockAddr) ? "✅" : "❌"}`);
  console.log(`   Timelock has PAUSER_ROLE:        ${await escrow.hasRole(PAUSER_ROLE, timelockAddr) ? "✅" : "❌"}`);
  console.log(`   Timelock has RELEASE_ROLE:       ${await escrow.hasRole(RELEASE_ROLE, timelockAddr) ? "✅" : "❌"}`);
  console.log(`   Deployer has DEFAULT_ADMIN_ROLE: ${await escrow.hasRole(DEFAULT_ADMIN_ROLE, deployer.address) ? "⚠️" : "✅ (renounced)"}`);

  console.log(`\n==================================================`);
  console.log(`✅ TRANSFER COMPLETE`);
  console.log(`==================================================`);
  console.log(`\n📋 NEXT STEPS:`);
  console.log(`1. Gnosis Safe must renounce DEFAULT_ADMIN_ROLE on Timelock:`);
  console.log(`   timelock.renounceRole(DEFAULT_ADMIN_ROLE, safeAddress)`);
  console.log(`2. All admin functions now require:`);
  console.log(`   a. Propose via Gnosis Safe → schedule()`);
  console.log(`   b. Wait ${parseInt(process.env.TIMELOCK_MIN_DELAY || "172800") / 86400} days`);
  console.log(`   c. Execute via Gnosis Safe → execute()`);
  console.log(`==================================================\n`);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("❌ Transfer failed:", error);
    process.exit(1);
  });
