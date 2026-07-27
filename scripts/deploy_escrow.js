const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  // Determine the target network from environment or hardhat argument
  const network = hre.network.name;
  const isMainnet = network === "polygon" || network === "polygonMainnet";
  const chainLabel = isMainnet ? "Polygon Mainnet" : "Polygon Amoy (Testnet)";
  const nativeCurrency = isMainnet ? "POL" : "POL (Test)";
  const explorerUrl = isMainnet ? "https://polygonscan.com" : "https://amoy.polygonscan.com";

  console.log("----------------------------------------------------");
  console.log(`Starting MedicalEscrow Deployment on ${chainLabel}...`);
  console.log(`Network: ${network} (Chain ID: ${hre.network.config.chainId})`);
  console.log("----------------------------------------------------");

  // 1. Validate Deployer Wallet
  const [deployer] = await hre.ethers.getSigners();
  const balance = await hre.ethers.provider.getBalance(deployer.address);

  console.log(`Deployer Address : ${deployer.address}`);
  console.log(`Deployer Balance : ${hre.ethers.formatEther(balance)} ${nativeCurrency}`);

  if (balance === 0n) {
    const faucetMsg = isMainnet
      ? "❌ Deployer wallet has 0 POL! Please fund the wallet with real POL from an exchange."
      : "❌ Deployer wallet has 0 POL! Please claim test tokens from the Polygon Amoy faucet:\n   https://faucet.polygon.technology/";
    throw new Error(faucetMsg);
  }

  // Warn on low balance for mainnet deployment
  if (isMainnet && balance < hre.ethers.parseEther("0.5")) {
    console.warn("⚠️  Low balance (< 0.5 POL). You may not have enough for gas on mainnet.");
  }

  // 2. Validate Oracle Address
  const oracleAddress = process.env.ESCROW_ORACLE_ADDRESS;
  if (!oracleAddress || !hre.ethers.isAddress(oracleAddress)) {
    throw new Error(
      "❌ Invalid or missing ESCROW_ORACLE_ADDRESS in environment! Please set a valid Ethereum address."
    );
  }

  console.log(`Oracle Address   : ${oracleAddress}`);

  // 3. Deploy MedicalEscrow Contract
  console.log("\nDeploying MedicalEscrow contract...");
  const MedicalEscrow = await hre.ethers.getContractFactory("MedicalEscrow");
  const escrow = await MedicalEscrow.deploy(oracleAddress);

  await escrow.waitForDeployment();
  const contractAddress = await escrow.getAddress();
  const deploymentTransaction = escrow.deploymentTransaction();
  const receipt = await deploymentTransaction.wait();
  const chainId = Number(hre.network.config.chainId || (await hre.ethers.provider.getNetwork()).chainId);

  // 4. Verify bytecode was deployed
  const runtimeBytecode = await hre.ethers.provider.getCode(contractAddress);
  if (runtimeBytecode === "0x") {
    throw new Error("❌ Deployment failed: No bytecode found at the contract address.");
  }

  // 5. Generate deployment artifact
  const deploymentMetadata = {
    contract_name: "MedicalEscrow",
    version: "2.0.0",
    abi_version: 1,
    network: network,
    chain_id: chainId,
    chain_label: chainLabel,
    contract_address: contractAddress,
    deployer_address: deployer.address,
    oracle_address: oracleAddress,
    transaction_hash: deploymentTransaction.hash,
    block_number: receipt.blockNumber,
    gas_used: receipt.gasUsed.toString(),
    runtime_bytecode_hash: hre.ethers.keccak256(runtimeBytecode),
    deployed_at_utc: new Date().toISOString(),
  };

  const deploymentSuffix = isMainnet ? "mainnet" : "amoy";
  const deploymentDirectory = path.join(__dirname, "..", "deployments");
  fs.mkdirSync(deploymentDirectory, { recursive: true });
  const metadataPath = path.join(deploymentDirectory, `deployment_${deploymentSuffix}.json`);
  fs.writeFileSync(metadataPath, `${JSON.stringify(deploymentMetadata, null, 2)}\n`, "utf8");

  console.log("----------------------------------------------------");
  console.log("✅ MedicalEscrow successfully deployed!");
  console.log(`Contract Address : ${contractAddress}`);
  console.log(`Transaction Hash : ${deploymentTransaction.hash}`);
  console.log(`Block Number     : ${receipt.blockNumber}`);
  console.log(`Gas Used         : ${receipt.gasUsed.toString()}`);
  console.log(`Deployment artifact saved to: ${metadataPath}`);
  console.log("----------------------------------------------------");
  console.log("\nNext Steps:");
  console.log(`1. Add to backend/.env: ESCROW_CONTRACT_ADDRESS=${contractAddress}`);
  console.log(`2. Add to frontend/.env.local: NEXT_PUBLIC_ESCROW_CONTRACT_ADDRESS=${contractAddress}`);
  console.log(`3. View on Explorer: ${explorerUrl}/address/${contractAddress}`);
  if (isMainnet) {
    console.log("\n🚀 MAINNET DEPLOYMENT — Additional Steps:");
    console.log("   - Verify contract on Polygonscan: npx hardhat verify --network polygon", contractAddress, oracleAddress);
    console.log("   - Transfer DEFAULT_ADMIN_ROLE to a Multi-Sig (Gnosis Safe)");
    console.log("   - Configure Sentry DSN, Slack webhooks, and dedicated RPC endpoints");
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
