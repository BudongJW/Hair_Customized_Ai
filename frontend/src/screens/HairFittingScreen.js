import * as ImagePicker from "expo-image-picker";
import React, { useEffect, useState } from "react";
import { Alert, Image, Pressable, StyleSheet, Text, View } from "react-native";
import { api } from "../api/client";
import { Panel } from "../components/Panel";
import { PrimaryButton } from "../components/PrimaryButton";
import { StatusPill } from "../components/StatusPill";
import { TextField } from "../components/TextField";
import { colors, radius, spacing } from "../theme/tokens";

export function HairFittingScreen({
  user,
  faceProfile,
  fittingJob,
  onBack,
  onFittingJobChanged,
  onFittingCompleted
}) {
  const [imageUri, setImageUri] = useState(null);
  const [fileName, setFileName] = useState("reference-hair.jpg");
  const [upload, setUpload] = useState(null);
  const [hairDesigns, setHairDesigns] = useState([]);
  const [selectedDesign, setSelectedDesign] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingDesigns, setLoadingDesigns] = useState(false);

  const canCreateJob = Boolean(
    faceProfile?.id && (
      selectedDesign?.id || (imageUri && fileName.trim())
    )
  );
  const jobCompleted = fittingJob?.status === "COMPLETED";

  useEffect(() => {
    loadHairDesigns();
  }, [user?.id]);

  const loadHairDesigns = async () => {
    if (!user?.id) {
      return;
    }

    try {
      setLoadingDesigns(true);
      const designs = await api.listHairDesigns(user.id);
      setHairDesigns(designs);
    } catch (error) {
      Alert.alert("헤어 디자인 불러오기 실패", error.message);
    } finally {
      setLoadingDesigns(false);
    }
  };

  const pickImage = async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert("사진 권한 필요", "헤어모델 사진을 선택하려면 사진 접근 권한이 필요합니다.");
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      allowsEditing: true,
      aspect: [4, 5],
      quality: 0.76
    });

    if (result.canceled) {
      return;
    }

    const asset = result.assets[0];
    setSelectedDesign(null);
    setImageUri(asset.uri);
    setFileName(asset.fileName || "reference-hair.jpg");
  };

  const selectDesign = (design) => {
    setSelectedDesign(design);
    setImageUri(null);
    setUpload(null);
  };

  const startFitting = async () => {
    if (!canCreateJob) {
      Alert.alert("헤어 사진 필요", "새 헤어모델 사진을 선택하거나 저장된 헤어 디자인을 선택해 주세요.");
      return;
    }

    try {
      setLoading(true);
      let payload = {
        profileId: faceProfile.id
      };

      if (selectedDesign?.id) {
        payload = {
          ...payload,
          hairDesignId: selectedDesign.id
        };
      } else {
        const uploadResponse = await api.createPresignedUrl({
          fileName,
          contentType: "image/jpeg",
          purpose: "REFERENCE_HAIR"
        });
        await api.uploadLocalImage(uploadResponse, imageUri, "image/jpeg");
        setUpload(uploadResponse);
        payload = {
          ...payload,
          referenceImageObjectKey: uploadResponse.objectKey
        };
      }

      const job = await api.createFittingJob(payload);
      onFittingJobChanged(job);
      Alert.alert("AI 피팅 요청 완료", "Python AI worker가 헤어마스크와 합성 결과를 생성합니다.");
    } catch (error) {
      Alert.alert("피팅 작업 생성 실패", error.message);
    } finally {
      setLoading(false);
    }
  };

  const refreshFittingJob = async () => {
    if (!fittingJob?.id) {
      return;
    }

    try {
      setLoading(true);
      const job = await api.getFittingJob(fittingJob.id);
      onFittingJobChanged(job);
      if (job.status === "COMPLETED") {
        await loadHairDesigns();
        onFittingCompleted(job);
      }
    } catch (error) {
      Alert.alert("상태 새로고침 실패", error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.wrap}>
      <View style={styles.headerRow}>
        <View style={styles.headerCopy}>
          <Text style={styles.title}>헤어모델 등록</Text>
          <Text style={styles.subtitle}>{user?.displayName}님의 얼굴에 적용할 참고 사진 또는 저장된 디자인</Text>
        </View>
        <StatusPill status={fittingJob?.status || "READY"} />
      </View>

      <Panel>
        <Text style={styles.sectionTitle}>새 헤어모델 사진</Text>
        <View style={styles.preview}>
          {imageUri ? (
            <Image source={{ uri: imageUri }} style={styles.image} />
          ) : (
            <View style={styles.placeholder}>
              <View style={styles.hairShape} />
              <View style={styles.faceShape} />
              <Text style={styles.placeholderText}>
                머리스타일이 잘 보이는 정면 인물 사진을 선택하세요.
              </Text>
            </View>
          )}
        </View>

        <TextField
          label="파일명"
          value={fileName}
          onChangeText={setFileName}
          placeholder="reference-hair.jpg"
        />

        <View style={styles.buttonRow}>
          <PrimaryButton label="사진 선택" variant="secondary" onPress={pickImage} />
          <PrimaryButton
            label={selectedDesign ? "선택한 디자인으로 피팅" : "피팅 작업 생성"}
            onPress={startFitting}
            loading={loading}
            disabled={!canCreateJob}
          />
        </View>
      </Panel>

      <Panel>
        <View style={styles.resultHeader}>
          <View style={styles.resultCopy}>
            <Text style={styles.sectionTitle}>저장된 헤어 디자인</Text>
            <Text style={styles.metaText}>
              분석이 완료된 헤어모델은 다음 피팅에서 다시 선택할 수 있습니다.
            </Text>
          </View>
          <PrimaryButton label="새로고침" variant="secondary" onPress={loadHairDesigns} loading={loadingDesigns} />
        </View>

        {hairDesigns.length > 0 ? (
          <View style={styles.designList}>
            {hairDesigns.map((design, index) => (
              <Pressable
                key={design.id}
                onPress={() => selectDesign(design)}
                style={[
                  styles.designItem,
                  selectedDesign?.id === design.id && styles.selectedDesign
                ]}
              >
                <View style={styles.designCopy}>
                  <Text style={styles.designTitle}>헤어 디자인 #{hairDesigns.length - index}</Text>
                  <Text style={styles.objectKey}>{design.referenceImageObjectKey}</Text>
                  {design.hairMaskObjectKey && (
                    <Text style={styles.metaText}>마스크 저장됨</Text>
                  )}
                </View>
                <StatusPill status={design.status} />
              </Pressable>
            ))}
          </View>
        ) : (
          <View style={styles.emptyBox}>
            <Text style={styles.emptyText}>아직 저장된 헤어 디자인이 없습니다.</Text>
          </View>
        )}
      </Panel>

      {fittingJob && (
        <Panel>
          <View style={styles.resultHeader}>
            <View style={styles.resultCopy}>
              <Text style={styles.sectionTitle}>최근 피팅 작업</Text>
              <Text style={styles.objectKey}>{fittingJob.referenceImageObjectKey}</Text>
              {fittingJob.hairMaskObjectKey && (
                <Text style={styles.objectKey}>마스크: {fittingJob.hairMaskObjectKey}</Text>
              )}
              {fittingJob.resultImageObjectKey && (
                <Text style={styles.objectKey}>결과: {fittingJob.resultImageObjectKey}</Text>
              )}
            </View>
            <StatusPill status={fittingJob.status} />
          </View>
          <PrimaryButton
            label={jobCompleted ? "결과 보기" : "AI 피팅 상태 새로고침"}
            onPress={refreshFittingJob}
            loading={loading}
          />
        </Panel>
      )}

      {upload && (
        <Panel style={styles.compactPanel}>
          <Text style={styles.metaLabel}>REFERENCE OBJECT</Text>
          <Text style={styles.objectKey}>{upload.objectKey}</Text>
        </Panel>
      )}

      <PrimaryButton label="뒤로" variant="secondary" onPress={onBack} />
    </View>
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
    fontWeight: "900",
    lineHeight: 33
  },
  subtitle: {
    color: colors.muted,
    fontSize: 13,
    fontWeight: "700",
    lineHeight: 19
  },
  sectionTitle: {
    color: colors.ink,
    fontSize: 18,
    fontWeight: "900"
  },
  preview: {
    backgroundColor: "#EEF2F6",
    borderRadius: radius.md,
    height: 300,
    overflow: "hidden"
  },
  image: {
    height: "100%",
    width: "100%"
  },
  placeholder: {
    alignItems: "center",
    flex: 1,
    justifyContent: "center",
    padding: spacing.md
  },
  hairShape: {
    backgroundColor: colors.ink,
    borderTopLeftRadius: 60,
    borderTopRightRadius: 60,
    height: 92,
    width: 116
  },
  faceShape: {
    backgroundColor: "#F3C9B8",
    borderRadius: 38,
    height: 76,
    marginTop: -28,
    width: 62
  },
  placeholderText: {
    color: colors.muted,
    fontSize: 13,
    fontWeight: "800",
    lineHeight: 19,
    marginTop: spacing.md,
    textAlign: "center"
  },
  buttonRow: {
    flexDirection: "row",
    gap: spacing.sm
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
  designList: {
    gap: spacing.sm
  },
  designItem: {
    alignItems: "flex-start",
    backgroundColor: "#FBFCFD",
    borderColor: colors.line,
    borderRadius: radius.md,
    borderWidth: 1,
    flexDirection: "row",
    gap: spacing.sm,
    justifyContent: "space-between",
    padding: spacing.md
  },
  selectedDesign: {
    backgroundColor: colors.softTeal,
    borderColor: colors.teal
  },
  designCopy: {
    flex: 1,
    gap: spacing.xs
  },
  designTitle: {
    color: colors.ink,
    fontSize: 15,
    fontWeight: "900"
  },
  objectKey: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: "700",
    lineHeight: 17
  },
  compactPanel: {
    gap: spacing.xs
  },
  metaLabel: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: "800"
  },
  metaText: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: "700",
    lineHeight: 17
  },
  emptyBox: {
    backgroundColor: "#EEF2F6",
    borderRadius: radius.md,
    padding: spacing.md
  },
  emptyText: {
    color: colors.muted,
    fontSize: 13,
    fontWeight: "800",
    textAlign: "center"
  }
});
