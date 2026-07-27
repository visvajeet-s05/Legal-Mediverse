const { ethers, network } = require("hardhat");

async function main() {
  console.log(`\n==================================================`);
  console.log(`Ownership Transfer — MedicalEscrow to Gnosis Safe`);
  console.log(`Network: ${network.name} (Chain ID: ${network.config.chainId})`);
  console.log(`==================================================\n`);

  const escrowContractAddress = process.env.ESCROW_CONTRACT_ADDRESS;
  const gnosisSafeAddress = process.env.GNOSIS_SAFE_ADDRESS;

  if (!escrowContractAddress || !gnosisSafeAddress) {
    throw new Error("Missing ESCROW_CONTRACT_ADDRESS or GNOSIS_SAFE_ADDRESS in .env");
  }

  const escrowAddress = ethers.getAddress(escrowContractAddress);
  const safeAddress = ethers.getAddress(gnosisSafeAddress);
  const [deployer] = await ethers.getSigners();

  console.log(`Deployer: ${deployer.address}`);
  console.log(`Contract: ${escrowAddress}`);
  console.log(`Gnosis Safe: ${safeAddress}\n`);

  const MedicalEscrow = await ethers.getContractFactory("MedicalEscrow");
  const escrow = MedicalEscrow.attach(escrowAddress);

  // Detect if contract uses AccessControl (hasRole) or Ownable (owner)
  let usesAccessControl = false;
  try {
    const DEFAULT_ADMIN_ROLE = "0x0000000000000000000000000000000000000000000000000000000000000000";
    const hasAdminRole = await escrow.hasRole(DEFAULT_ADMIN_ROLE, deployer.address);
    usesAccessControl = true;
    console.log("Contract uses AccessControl (DEFAULT_ADMIN_ROLE)");

    if (!hasAdminRole) {
      throw new Error(`Deployer does not have DEFAULT_ADMIN_ROLE!`);
    }

    // Grant DEFAULT_ADMIN_ROLE to Gnosis Safe
    const tx = await escrow.grantRole(DEFAULT_ADMIN_ROLE, safeAddress, {
      maxPriorityFeePerGas: ethers.parseUnits("35", "gwei"),
    });
    console.log(`Granting DEFAULT_ADMIN_ROLE... Tx: ${tx.hash}`);
    await tx.wait(2);
    console.log("DEFAULT_ADMIN_ROLE granted to Gnosis Safe!");

    // Also grant PAUSER_ROLE for emergency pause capability
    const PAUSER_ROLE = await escrow.PAUSER_ROLE();
    const tx2 = await escrow.grantRole(PAUSER_ROLE, safeAddress, {
      maxPriorityFeePerGas: ethers.parseUnits("35", "gwei"),
    });
    console.log(`Granting PAUSER_ROLE... Tx: ${tx2.hash}`);
    await tx2.wait(2);
    console.log("PAUSER_ROLE granted to Gnosis Safe!");

    // Also grant RELEASE_ROLE for funds release
    const RELEASE_ROLE = await escrow.RELEASE_ROLE();
    const tx3 = await escrow.grantRole(RELEASE_ROLE, safeAddress, {
      maxPriorityFeePerGas: ethers.parseUnits("35", "gwei"),
    });
    console.log(`Granting RELEASE_ROLE... Tx: ${tx3.hash}`);
    await tx3.wait(2);
    console.log("RELEASE_ROLE granted to Gnosis Safe!");

    console.log("\nAll administrative roles transferred to Gnosis Safe!");
    console.log("Consider renouncing roles from deployer after verification.");
    return;
  } catch (e) {
    if (usesAccessControl) throw e;
    // Fall through to Ownable path
  }

  // Ownable path
  const currentOwner = await escrow.owner();
  console.log(`Current Owner: ${currentOwner}`);

  if (currentOwner.toLowerCase() !== deployer.address.toLowerCase()) {
    throw new Error("Deployer is not the current owner!");
  }

  if (currentOwner.toLowerCase() === safeAddress.toLowerCase()) {
    console.log("Ownership already set to Gnosis Safe.");
    return;
  }

  const tx = await escrow.transferOwnership(safeAddress, {
    maxPriorityFeePerGas: ethers.parseUnits("35", "gwei"),
  });
  console.log(`Transfer tx sent: ${tx.hash}`);
  await tx.wait(2);
  console.log("Ownership transferred successfully!");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("Transfer failed:", error);
    process.exit(1);
  });
