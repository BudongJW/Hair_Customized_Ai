import React from "react";
import { ActivityIndicator, Image, StyleSheet, Text, View } from "react-native";
import { api } from "../api/client";
import { Panel } from "../components/Panel";
import { PrimaryButton } from "../components/PrimaryButton";
import { StatusPill } from "../components/StatusPill";
import { colors, radius, spacing } from "../theme/tokens";

export function ResultScreen({
  user,
  faceProfile,
  fittingJob,
  onTryAnother,
  onReset
}) {
  const [imageUrls, setImageUrls] = React.useState({});
  const [loadingImages, setLoadingImages] = React.useState(false);
  const [previewError, setPreviewError] = React.useState(null);
  const hasResult = fittingJob?.status === "COMPLETED" && fittingJob?.resultImageObjectKey;
  const isPrepared = fittingJob?.status === "PREPARED";

  React.useEffect(() => {
    let mounted = true;

    async function loadImageUrls() {
      const keys = {
        face: faceProfile?.originalImageObjectKey,
        reference: fittingJob?.referenceImageObjectKey,
        result: fittingJob?.resultImageObjectKey,
        mask: fittingJob?.hairMaskObjectKey,
        layer: fittingJob?.hairLayerObjectKey,
        targetMask: fittingJob?.targetHairMaskObjectKey,
        inpaintingMask: fittingJob?.inpaintingMaskObjectKey,
        faceProtection: fittingJob?.faceProtectionMaskObjectKey
      };
      if (fittingJob?.status === "COMPLETED" && fittingJob?.id) {
        keys.baldCanvas = `ai/fitting-jobs/${fittingJob.id}/bald-canvas.png`;
      }
      const entries = Object.entries(keys).filter(([, objectKey]) => Boolean(objectKey));
      setImageUrls({});
      setPreviewError(null);

      if (entries.length === 0) {
        setImageUrls({});
        setLoadingImages(false);
        return;
      }

      setLoadingImages(true);
      setPreviewError(null);

      try {
        const resolved = await Promise.allSettled(
          entries.map(async ([name, objectKey]) => {
            const response = await api.createPresignedDownloadUrl({ objectKey });
            return [name, response.downloadUrl];
          })
        );

        if (mounted) {
          setImageUrls(Object.fromEntries(
            resolved.filter((entry) => entry.status === "fulfilled").map((entry) => entry.value)
          ));
          if (resolved.some((entry) => entry.status === "rejected")) {
            setPreviewError("일부 이미지를 불러오지 못했습니다. 화면을 다시 열어 주세요.");
          }
        }
      } catch (error) {
        if (mounted) {
          setPreviewError(error.message);
          setImageUrls({});
        }
      } finally {
        if (mounted) {
          setLoadingImages(false);
        }
      }
    }

    loadImageUrls();

    return () => {
      mounted = false;
    };
  }, [
    faceProfile?.originalImageObjectKey,
    fittingJob?.referenceImageObjectKey,
    fittingJob?.resultImageObjectKey,
    fittingJob?.hairMaskObjectKey,
    fittingJob?.hairLayerObjectKey,
    fittingJob?.targetHairMaskObjectKey,
    fittingJob?.inpaintingMaskObjectKey,
    fittingJob?.faceProtectionMaskObjectKey,
    fittingJob?.id,
    fittingJob?.status
  ]);

  return (
    <View style={styles.wrap}>
      <View style={styles.headerRow}>
        <View style={styles.headerCopy}>
          <Text style={styles.title}>피팅 결과</Text>
          <Text style={styles.subtitle}>{user?.displayName}님의 헤어 합성 작업</Text>
        </View>
        <StatusPill status={fittingJob?.status || "READY"} />
      </View>

      {loadingImages && (
        <Panel style={styles.loadingPanel}>
          <ActivityIndicator />
          <Text style={styles.loadingText}>결과 이미지를 불러오는 중입니다.</Text>
        </Panel>
      )}

      {previewError && (
        <Panel>
          <Text style={styles.errorTitle}>이미지 미리보기 실패</Text>
          <Text style={styles.errorText}>{previewError}</Text>
        </Panel>
      )}

      <View style={styles.compareRow}>
        <PreviewPanel title="내 얼굴" imageUrl={imageUrls.face} />
        <PreviewPanel title="헤어모델 사진" imageUrl={imageUrls.reference} />
      </View>

      <Panel>
        <Text style={styles.sectionTitle}>
          {hasResult ? "완료된 결과" : isPrepared ? "합성 준비 완료" : fittingJob?.status === "FAILED" ? "처리 실패" : "처리 중"}
        </Text>
        {hasResult && imageUrls.result ? (
          <Image source={{ uri: imageUrls.result }} style={styles.resultImage} />
        ) : (
          <View style={styles.emptyResult}>
            <Text style={styles.emptyText}>
              {isPrepared
                ? "사진 분석과 마스크 저장이 완료되었습니다. 아직 합성 이미지는 생성되지 않았습니다."
                : fittingJob?.status === "FAILED"
                  ? fittingJob.failureReason || "사진을 처리하지 못했습니다."
                  : "결과 이미지를 준비하고 있습니다."}
            </Text>
          </View>
        )}
        <Text style={styles.objectKey}>참고 사진: {fittingJob?.referenceImageObjectKey || "-"}</Text>
        <Text style={styles.objectKey}>결과 이미지: {fittingJob?.resultImageObjectKey || "-"}</Text>
        <Text style={styles.meta}>얼굴 프로필 ID: {fittingJob?.profileId || faceProfile?.id || "-"}</Text>
        {fittingJob?.hairDesignId && (
          <Text style={styles.meta}>저장된 헤어 디자인 ID: {fittingJob.hairDesignId}</Text>
        )}
      </Panel>

      {(imageUrls.mask || imageUrls.layer) && (
        <View style={styles.debugSection}>
          <Text style={styles.sectionTitle}>헤어 분석 디버그</Text>
          <View style={styles.compareRow}>
            <PreviewPanel title="헤어마스크" imageUrl={imageUrls.mask} />
            <PreviewPanel title="분리된 헤어레이어" imageUrl={imageUrls.layer} />
          </View>
          <Text style={styles.meta}>마스크와 레이어를 보면 머리카락이 어디까지 잘렸는지 확인할 수 있습니다.</Text>
        </View>
      )}

      {(imageUrls.targetMask || imageUrls.inpaintingMask || imageUrls.faceProtection) && (
        <View style={styles.debugSection}>
          <Text style={styles.sectionTitle}>사용자 머리 분석</Text>
          <View style={styles.compareRow}>
            <PreviewPanel title="기존 머리 영역" imageUrl={imageUrls.targetMask} />
            <PreviewPanel title="합성 수정 영역" imageUrl={imageUrls.inpaintingMask} />
          </View>
          <View style={styles.compareRow}>
            <PreviewPanel title="얼굴 보호 영역" imageUrl={imageUrls.faceProtection} />
            <PreviewPanel title="머리 제거 결과" imageUrl={imageUrls.baldCanvas} />
          </View>
        </View>
      )}

      <View style={styles.buttonRow}>
        <PrimaryButton label="다른 사진 등록" variant="secondary" onPress={onTryAnother} />
        <PrimaryButton label="홈으로" onPress={onReset} />
      </View>
    </View>
  );
}

function PreviewPanel({ title, imageUrl }) {
  return (
    <Panel style={styles.comparePanel}>
      <Text style={styles.compareLabel}>{title}</Text>
      {imageUrl ? (
        <Image source={{ uri: imageUrl }} style={styles.previewImage} />
      ) : (
        <View style={styles.previewFallback}>
          <Text style={styles.emptyText}>이미지 없음</Text>
        </View>
      )}
    </Panel>
  );
}

const styles = StyleSheet.create({
  wrap: {
    gap: spacing.md
  },
  headerRow: {
    alignItems: "flex-start",
    flexDirection: "row",
    gap: spacing.md,
    justifyContent: "space-between"
  },
  headerCopy: {
    flex: 1,
    gap: spacing.xs
  },
  title: {
    color: colors.ink,
    fontSize: 26,
    fontWeight: "900"
  },
  subtitle: {
    color: colors.muted,
    fontSize: 14,
    fontWeight: "700"
  },
  loadingPanel: {
    alignItems: "center",
    flexDirection: "row",
    gap: spacing.sm
  },
  loadingText: {
    color: colors.muted,
    fontSize: 13,
    fontWeight: "700"
  },
  errorTitle: {
    color: colors.danger,
    fontSize: 16,
    fontWeight: "900"
  },
  errorText: {
    color: colors.muted,
    fontSize: 13,
    fontWeight: "700",
    lineHeight: 19
  },
  compareRow: {
    flexDirection: "row",
    gap: spacing.sm
  },
  comparePanel: {
    flex: 1,
    minWidth: 0,
    minHeight: 230
  },
  debugSection: {
    gap: spacing.sm
  },
  compareLabel: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: "900"
  },
  previewImage: {
    aspectRatio: 4 / 5,
    backgroundColor: "#EEF2F6",
    borderRadius: radius.md,
    width: "100%"
  },
  previewFallback: {
    alignItems: "center",
    aspectRatio: 4 / 5,
    backgroundColor: "#EEF2F6",
    borderRadius: radius.md,
    justifyContent: "center",
    width: "100%"
  },
  sectionTitle: {
    color: colors.ink,
    fontSize: 18,
    fontWeight: "900"
  },
  resultImage: {
    aspectRatio: 4 / 5,
    backgroundColor: "#EEF2F6",
    borderRadius: radius.md,
    width: "100%"
  },
  emptyResult: {
    alignItems: "center",
    aspectRatio: 4 / 5,
    backgroundColor: "#EEF2F6",
    borderRadius: radius.md,
    justifyContent: "center",
    padding: spacing.md,
    width: "100%"
  },
  emptyText: {
    color: colors.muted,
    fontSize: 13,
    fontWeight: "800",
    lineHeight: 19,
    textAlign: "center"
  },
  objectKey: {
    color: colors.ink,
    fontSize: 13,
    fontWeight: "800",
    lineHeight: 18
  },
  meta: {
    color: colors.muted,
    fontSize: 12,
    lineHeight: 17
  },
  buttonRow: {
    flexDirection: "row",
    gap: spacing.sm
  }
});
