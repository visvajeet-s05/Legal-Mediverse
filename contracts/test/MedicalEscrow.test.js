const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("MedicalEscrow", function () {
  const targetAmount = ethers.parseEther("1");
  const donationAmount = ethers.parseEther("0.5");

  let escrow;
  let admin;
  let donor;
  let hospital;
  let oracle;
  let releaseOperator;
  let unauthorized;

  async function createCampaign() {
    return escrow.connect(admin).createCampaign(hospital.address, targetAmount);
  }

  beforeEach(async function () {
    [admin, donor, hospital, oracle, releaseOperator, unauthorized] = await ethers.getSigners();
    const MedicalEscrow = await ethers.getContractFactory("MedicalEscrow");
    escrow = await MedicalEscrow.deploy(oracle.address, ethers.ZeroHash, 0);
    await escrow.waitForDeployment();
    await escrow.grantRole(await escrow.RELEASE_ROLE(), releaseOperator.address);
  });

  describe("campaign management", function () {
    it("creates a campaign with a pending verification state", async function () {
      await expect(createCampaign())
        .to.emit(escrow, "CampaignCreated")
        .withArgs(1, admin.address, hospital.address, targetAmount);

      const campaign = await escrow.campaigns(1);
      expect(campaign.creator).to.equal(admin.address);
      expect(campaign.hospitalWallet).to.equal(hospital.address);
      expect(campaign.targetAmount).to.equal(targetAmount);
      expect(campaign.verificationStatus).to.equal(0);
    });

    it("rejects a zero hospital address and a zero target", async function () {
      await expect(
        escrow.createCampaign(ethers.ZeroAddress, targetAmount),
      ).to.be.revertedWithCustomError(escrow, "InvalidHospitalAddress");
      await expect(
        escrow.createCampaign(hospital.address, 0),
      ).to.be.revertedWithCustomError(escrow, "InvalidTargetAmount");
    });
  });

  describe("donations", function () {
    it("accounts for native-token donations and emits the receipt event", async function () {
      await createCampaign();
      await expect(escrow.connect(donor).donate(1, { value: donationAmount }))
        .to.emit(escrow, "DonationReceived")
        .withArgs(1, donor.address, donationAmount);

      const campaign = await escrow.campaigns(1);
      expect(campaign.amountRaised).to.equal(donationAmount);
      expect(await escrow.contributions(1, donor.address)).to.equal(donationAmount);
    });

    it("rejects a zero-value donation", async function () {
      await createCampaign();
      await expect(escrow.connect(donor).donate(1, { value: 0 }))
        .to.be.revertedWithCustomError(escrow, "InvalidDonationAmount");
    });
  });

  describe("oracle verification and release", function () {
    it("limits verification submission to ORACLE_ROLE", async function () {
      await createCampaign();
      await expect(
        escrow.connect(unauthorized).fulfillBillVerification(1, true, targetAmount, 0),
      )
        .to.be.revertedWithCustomError(escrow, "AccessControlUnauthorizedAccount")
        .withArgs(unauthorized.address, await escrow.ORACLE_ROLE());

      await expect(escrow.connect(oracle).fulfillBillVerification(1, true, targetAmount, 0))
        .to.emit(escrow, "BillVerificationFulfilled")
        .withArgs(1, true, targetAmount, 0);
    });

    it("requires RELEASE_ROLE and pays the nominated hospital exactly once", async function () {
      await createCampaign();
      await escrow.connect(donor).donate(1, { value: targetAmount });
      await escrow.connect(oracle).fulfillBillVerification(1, true, targetAmount, 0);

      await expect(escrow.connect(unauthorized).releaseFunds(1))
        .to.be.revertedWithCustomError(escrow, "AccessControlUnauthorizedAccount");

      const hospitalBalanceBefore = await ethers.provider.getBalance(hospital.address);
      await expect(escrow.connect(releaseOperator).releaseFunds(1))
        .to.emit(escrow, "FundsReleased")
        .withArgs(1, hospital.address, targetAmount);
      const hospitalBalanceAfter = await ethers.provider.getBalance(hospital.address);
      expect(hospitalBalanceAfter - hospitalBalanceBefore).to.equal(targetAmount);

      await expect(escrow.connect(releaseOperator).releaseFunds(1))
        .to.be.revertedWithCustomError(escrow, "CampaignAlreadyReleased")
        .withArgs(1);
    });

    it("does not release when the bill total or fraud score fails the policy", async function () {
      await createCampaign();
      await escrow.connect(donor).donate(1, { value: donationAmount });
      await escrow.connect(oracle).fulfillBillVerification(1, true, targetAmount - 1n, 10);

      await expect(escrow.connect(releaseOperator).releaseFunds(1))
        .to.be.revertedWithCustomError(escrow, "CampaignDoesNotMeetReleaseRequirements")
        .withArgs(1);
    });
  });

  describe("emergency controls and refunds", function () {
    it("pauses donations, verification, and releases until unpaused", async function () {
      await createCampaign();
      await escrow.pause();
      await expect(escrow.connect(donor).donate(1, { value: donationAmount })).to.be.revertedWithCustomError(
        escrow,
        "EnforcedPause",
      );
      await escrow.unpause();
      await escrow.connect(donor).donate(1, { value: donationAmount });
    });

    it("allows refunds only after an explicit oracle rejection", async function () {
      await createCampaign();
      await escrow.connect(donor).donate(1, { value: donationAmount });
      await expect(escrow.connect(donor).claimRefund(1))
        .to.be.revertedWithCustomError(escrow, "CampaignNotRejected")
        .withArgs(1);

      await escrow.connect(oracle).fulfillBillVerification(1, false, 0, 100);
      await expect(escrow.connect(donor).claimRefund(1))
        .to.changeEtherBalances([escrow, donor], [-donationAmount, donationAmount]);
    });
  });
});
