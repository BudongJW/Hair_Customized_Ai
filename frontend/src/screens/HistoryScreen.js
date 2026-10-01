import React from "react";
import { StyleSheet, Text, View } from "react-native";
import { Panel } from "../components/Panel";
import { PrimaryButton } from "../components/PrimaryButton";
import { StatusPill } from "../components/StatusPill";
import { colors, spacing } from "../theme/tokens";

export function HistoryScreen({ user, fittingJobs = [], onBack, onOpenResult }) {
  const completedJobs = fittingJobs.filter((job) => job.status === "COMPLETED");

  return (
    <View style={styles.wrap}>
      <View style={styles.headerCopy}>
        <Text style={styles.title}>예전 결과물</Text>
        <Text style={styles.subtitle}>
          {user.displayName}님의 피팅 기록 {fittingJobs.length}개
        </Text>
      </View>

      {fittingJobs.length > 0 ? (
        <View style={styles.list}>
          {fittingJobs.map((job, index) => (
            <Panel key={job.id}>
              <View style={styles.resultHeader}>
                <View style={styles.resultCopy}>
                  <Text style={styles.resultTitle}>피팅 작업 #{fittingJobs.length - index}</Text>
                  <Text style={styles.resultMeta}>참고 사진: {job.referenceImageObjectKey}</Text>
                  {job.resultImageObjectKey && (
                    <Text style={styles.resultMeta}>결과 이미지: {job.resultImageObjectKey}</Text>
                  )}
                </View>
                <StatusPill status={job.status} />
              </View>
              <PrimaryButton
                label={job.status === "PREPARED" ? "준비 결과 보기" : job.status === "COMPLETED" ? "결과 보기" : "처리 상태 보기"}
                onPress={() => onOpenResult(job)}
              />
            </Panel>
          ))}
        </View>
      ) : (
        <Panel>
          <Text style={styles.emptyTitle}>아직 저장된 결과물이 없습니다.</Text>
          <Text style={styles.emptyBody}>
            헤어스타일 모델 사진을 등록하고 AI 피팅을 완료하면 여기에 표시됩니다.
          </Text>
        </Panel>
      )}

      {completedJobs.length > 0 && (
        <Text style={styles.summaryText}>완료된 결과 {completedJobs.length}개를 확인할 수 있습니다.</Text>
      )}

      <PrimaryButton label="홈으로" variant="secondary" onPress={onBack} />
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    gap: spacing.md
  },
  headerCopy: {
    gap: spacing.xs,
    paddingTop: spacing.sm
  },
  title: {
    color: colors.ink,
    fontSize: 28,
    fontWeight: "900"
  },
  subtitle: {
    color: colors.muted,
    fontSize: 14,
    fontWeight: "700"
  },
  list: {
    gap: spacing.md
  },
  resultHeader: {
    alignItems: "flex-start",
    flexDirection: "row",
    gap: spacing.md,
    justifyContent: "space-between"
  },
  resultCopy: {
    flex: 1,
    gap: spacing.xs
  },
  resultTitle: {
    color: colors.ink,
    fontSize: 18,
    fontWeight: "900"
  },
  resultMeta: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: "700",
    lineHeight: 17
  },
  emptyTitle: {
    color: colors.ink,
    fontSize: 18,
    fontWeight: "900"
  },
  emptyBody: {
    color: colors.muted,
    fontSize: 14,
    fontWeight: "600",
    lineHeight: 20
  },
  summaryText: {
    color: colors.muted,
    fontSize: 13,
    fontWeight: "800",
    textAlign: "center"
  }
});
