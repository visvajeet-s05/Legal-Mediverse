const hre = require("hardhat");

async function main() {
  console.log("Deploying MedicalEscrow contract to Polygon Amoy...");

  const oracleAddress = process.env.ESCROW_ORACLE_ADDRESS;
  if (!/^0x[0-9a-fA-F]{40}$/.test(oracleAddress || "") || /^0x0{40}$/i.test(oracleAddress)) {
    throw new Error("Set ESCROW_ORACLE_ADDRESS to the non-zero account that will submit audited verification results.");
  }

  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying with account:", deployer.address);

  const MedicalEscrow = await hre.ethers.getContractFactory("MedicalEscrow");
  
  const jobId = process.env.CHAINLINK_JOB_ID || hre.ethers.ZeroHash;
  const fee = BigInt(process.env.CHAINLINK_FEE_WEI || "0");

  // Pass required constructor arguments
  const escrow = await MedicalEscrow.deploy(oracleAddress, jobId, fee);

  await escrow.waitForDeployment();
  const contractAddress = await escrow.getAddress();

  console.log("==================================================");
  console.log("MedicalEscrow deployed successfully!");
  console.log("Contract Address:", contractAddress);
  console.log("==================================================");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
