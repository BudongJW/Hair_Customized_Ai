import React from "react";
import { StyleSheet, Text, View } from "react-native";
import { colors, radius, spacing } from "../theme/tokens";

export function StepRail({ steps, activeKey }) {
  const activeIndex = steps.findIndex((step) => step.key === activeKey);
  return (
    <View style={styles.rail}>
      {steps.map((step, index) => {
        const active = index <= activeIndex;
        return (
          <View key={step.key} style={styles.item}>
            <View style={[styles.dot, active && styles.activeDot]} />
            <Text style={[styles.label, active && styles.activeLabel]}>{step.label}</Text>
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  rail: {
    backgroundColor: colors.panel,
    borderColor: colors.line,
    borderRadius: radius.md,
    borderWidth: 1,
    flexDirection: "row",
    justifyContent: "space-between",
    padding: spacing.sm
  },
  item: {
    alignItems: "center",
    flex: 1,
    gap: spacing.xs
  },
  dot: {
    backgroundColor: "#D9DEE7",
    borderRadius: 5,
    height: 10,
    width: 10
  },
  activeDot: {
    backgroundColor: colors.coral
  },
  label: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: "700"
  },
  activeLabel: {
    color: colors.ink
  }
});
