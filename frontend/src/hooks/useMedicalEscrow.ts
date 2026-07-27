"use client";

import { useMemo, useCallback } from "react";
import {
  useAccount,
  useWriteContract,
  useWaitForTransactionReceipt,
  useWatchContractEvent,
  useReadContract,
} from "wagmi";
import {
  type Address,
  type Hash,
  parseEther,
} from "viem";
import { MEDICAL_ESCROW_ABI, MEDICAL_ESCROW_CONTRACT_ADDRESS } from "../lib/contracts";

const CONTRACT_CONFIG = {
  address: MEDICAL_ESCROW_CONTRACT_ADDRESS as `0x${string}`,
  abi: MEDICAL_ESCROW_ABI,
} as const;

// ─── Typed Log Interfaces ─────────────────────────────────────────────────

export interface CampaignCreatedLog {
  campaignId: bigint;
  creator: Address;
  hospital: Address;
  targetAmount: bigint;
}

export interface DonationReceivedLog {
  campaignId: bigint;
  donor: Address;
  amount: bigint;
}

export interface BillVerificationRequestedLog {
  campaignId: bigint;
  requestId: Hash;
}

export interface BillVerificationFulfilledLog {
  campaignId: bigint;
  isVerified: boolean;
  billTotal: bigint;
  fraudRiskScore: bigint;
}

export interface FundsReleasedLog {
  campaignId: bigint;
  hospital: Address;
  amount: bigint;
}

export interface RefundIssuedLog {
  campaignId: bigint;
  donor: Address;
  amount: bigint;
}

// ─── Campaign Data Shape ──────────────────────────────────────────────────

export interface CampaignData {
  creator: Address;
  hospitalWallet: Address;
  targetAmount: bigint;
  amountRaised: bigint;
  billTotalExtracted: bigint;
  fraudRiskScore: bigint;
  verificationStatus: "Pending" | "Approved" | "Rejected";
  isReleased: boolean;
}

// ─── Hook ─────────────────────────────────────────────────────────────────

export function useMedicalEscrow() {
  const { isConnected, address } = useAccount();
  const { data: hash, isPending, writeContract } = useWriteContract();
  const { isLoading: isConfirming, isSuccess: isConfirmed } =
    useWaitForTransactionReceipt({ hash });

  // ── Write Contract Actions ───────────────────────────────────────────

  const createCampaign = useCallback(
    async (hospitalWallet: Address, targetAmount: bigint) => {
      if (!isConnected || !address) throw new Error("Connect a wallet first.");

      return writeContract({
        ...CONTRACT_CONFIG,
        functionName: "createCampaign",
        args: [hospitalWallet, targetAmount],
      });
    },
    [address, isConnected, writeContract],
  );

  const donate = useCallback(
    async (campaignId: bigint, amount: bigint) => {
      if (!isConnected || !address) throw new Error("Connect a wallet first.");

      return writeContract({
        ...CONTRACT_CONFIG,
        functionName: "donate",
        args: [campaignId],
        value: amount,
      });
    },
    [address, isConnected, writeContract],
  );

  const requestBillVerification = useCallback(
    async (campaignId: bigint, billReference: string) => {
      if (!isConnected || !address) throw new Error("Connect a wallet first.");

      return writeContract({
        ...CONTRACT_CONFIG,
        functionName: "requestBillVerification",
        args: [campaignId, billReference],
      });
    },
    [address, isConnected, writeContract],
  );

  const fulfillBillVerification = useCallback(
    async (
      campaignId: bigint,
      isVerified: boolean,
      billTotalExtracted: bigint,
      fraudRiskScore: bigint,
    ) => {
      if (!isConnected || !address) throw new Error("Connect a wallet first.");

      return writeContract({
        ...CONTRACT_CONFIG,
        functionName: "fulfillBillVerification",
        args: [campaignId, isVerified, billTotalExtracted, fraudRiskScore],
      });
    },
    [address, isConnected, writeContract],
  );

  /**
   * Convenience wrapper that combines requestBillVerification +
   * fulfillBillVerification into a single verifyBill flow.
   * The oracle submits the OCR result directly via fulfillBillVerification.
   */
  const verifyBill = useCallback(
    async (
      campaignId: bigint,
      isVerified: boolean,
      billTotalExtracted: bigint,
      fraudRiskScore: bigint,
      billReference: string = "",
    ) => {
      if (!isConnected || !address) throw new Error("Connect a wallet first.");

      // Step 1: Request verification (emits auditable request marker)
      if (billReference) {
        await writeContract({
          ...CONTRACT_CONFIG,
          functionName: "requestBillVerification",
          args: [campaignId, billReference],
        });
      }

      // Step 2: Fulfill verification with OCR result
      return writeContract({
        ...CONTRACT_CONFIG,
        functionName: "fulfillBillVerification",
        args: [campaignId, isVerified, billTotalExtracted, fraudRiskScore],
      });
    },
    [address, isConnected, writeContract],
  );

  const releaseFunds = useCallback(
    async (campaignId: bigint) => {
      if (!isConnected || !address) throw new Error("Connect a wallet first.");

      return writeContract({
        ...CONTRACT_CONFIG,
        functionName: "releaseFunds",
        args: [campaignId],
      });
    },
    [address, isConnected, writeContract],
  );

  const claimRefund = useCallback(
    async (campaignId: bigint) => {
      if (!isConnected || !address) throw new Error("Connect a wallet first.");

      return writeContract({
        ...CONTRACT_CONFIG,
        functionName: "claimRefund",
        args: [campaignId],
      });
    },
    [address, isConnected, writeContract],
  );

  // ── Read Contract State ──────────────────────────────────────────────

  function useCampaign(campaignId: bigint) {
    const { data, isLoading, refetch } = useReadContract({
      ...CONTRACT_CONFIG,
      functionName: "campaigns",
      args: [campaignId],
    });

    return useMemo(() => {
      if (!data) return { campaign: null, isLoading, refetch };

      const [
        creator,
        hospitalWallet,
        targetAmount,
        amountRaised,
        billTotalExtracted,
        fraudRiskScore,
        verificationStatus,
        isReleased,
      ] = data as readonly [
        Address,
        Address,
        bigint,
        bigint,
        bigint,
        bigint,
        number,
        boolean,
      ];

      const campaign: CampaignData = {
        creator,
        hospitalWallet,
        targetAmount,
        amountRaised,
        billTotalExtracted,
        fraudRiskScore,
        verificationStatus: ["Pending", "Approved", "Rejected"][
          verificationStatus
        ] as "Pending" | "Approved" | "Rejected",
        isReleased,
      };

      return { campaign, isLoading, refetch };
    }, [data, isLoading, refetch]);
  }

  function useCampaignCount() {
    const { data, isLoading, refetch } = useReadContract({
      ...CONTRACT_CONFIG,
      functionName: "campaignCount",
    });

    return {
      count: data ? Number(data) : 0,
      isLoading,
      refetch,
    };
  }

  function useUserContribution(
    campaignId: bigint,
    userAddress: Address,
  ) {
    const { data, isLoading } = useReadContract({
      ...CONTRACT_CONFIG,
      functionName: "contributions",
      args: [campaignId, userAddress],
    });

    return {
      contribution: data ? (data as bigint) : 0n,
      isLoading,
    };
  }

  // ── Event Watchers ────────────────────────────────────────────────────

  function useWatchCampaignCreated(
    onLogs: (logs: CampaignCreatedLog[]) => void,
  ) {
    return useWatchContractEvent({
      ...CONTRACT_CONFIG,
      eventName: "CampaignCreated",
      onLogs,
    });
  }

  function useWatchDonationReceived(
    onLogs: (logs: DonationReceivedLog[]) => void,
  ) {
    return useWatchContractEvent({
      ...CONTRACT_CONFIG,
      eventName: "DonationReceived",
      onLogs,
    });
  }

  function useWatchBillVerificationRequested(
    onLogs: (logs: BillVerificationRequestedLog[]) => void,
  ) {
    return useWatchContractEvent({
      ...CONTRACT_CONFIG,
      eventName: "BillVerificationRequested",
      onLogs,
    });
  }

  function useWatchBillVerificationFulfilled(
    onLogs: (logs: BillVerificationFulfilledLog[]) => void,
  ) {
    return useWatchContractEvent({
      ...CONTRACT_CONFIG,
      eventName: "BillVerificationFulfilled",
      onLogs,
    });
  }

  function useWatchFundsReleased(
    onLogs: (logs: FundsReleasedLog[]) => void,
  ) {
    return useWatchContractEvent({
      ...CONTRACT_CONFIG,
      eventName: "FundsReleased",
      onLogs,
    });
  }

  function useWatchRefundIssued(
    onLogs: (logs: RefundIssuedLog[]) => void,
  ) {
    return useWatchContractEvent({
      ...CONTRACT_CONFIG,
      eventName: "RefundIssued",
      onLogs,
    });
  }

  // ── Composed Return ───────────────────────────────────────────────────

  return useMemo(
    () => ({
      // State
      isConnected,
      address,
      hash,
      isPending,
      isConfirming,
      isConfirmed,

      // Write Actions
      createCampaign,
      donate,
      requestBillVerification,
      fulfillBillVerification,
      verifyBill,
      releaseFunds,
      claimRefund,

      // Read Hooks
      useCampaign,
      useCampaignCount,
      useUserContribution,

      // Event Watchers
      useWatchCampaignCreated,
      useWatchDonationReceived,
      useWatchBillVerificationRequested,
      useWatchBillVerificationFulfilled,
      useWatchFundsReleased,
      useWatchRefundIssued,
    }),
    [
      address,
      hash,
      isConnected,
      isConfirming,
      isConfirmed,
      isPending,
      createCampaign,
      donate,
      requestBillVerification,
      fulfillBillVerification,
      verifyBill,
      releaseFunds,
      claimRefund,
      useCampaign,
      useCampaignCount,
      useUserContribution,
      useWatchCampaignCreated,
      useWatchDonationReceived,
      useWatchBillVerificationRequested,
      useWatchBillVerificationFulfilled,
      useWatchFundsReleased,
      useWatchRefundIssued,
    ],
  );
}
