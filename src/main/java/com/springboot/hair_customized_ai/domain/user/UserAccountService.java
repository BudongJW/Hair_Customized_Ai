package com.springboot.hair_customized_ai.domain.user;

import jakarta.persistence.EntityNotFoundException;
import java.util.Locale;
import java.util.UUID;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

@Service
@RequiredArgsConstructor
public class UserAccountService {

    private final UserAccountRepository userAccountRepository;
    private final PasswordEncoder passwordEncoder;

    @Transactional
    public UserAccount create(String email, String displayName) {
        String normalizedEmail = normalizeEmail(email);
        if (userAccountRepository.existsByEmailIgnoreCase(normalizedEmail)) {
            throw new ResponseStatusException(HttpStatus.CONFLICT, "Email already exists");
        }

        return userAccountRepository.save(new UserAccount(normalizedEmail, displayName));
    }

    @Transactional
    public UserAccount registerWithPassword(String email, String displayName, String password) {
        String normalizedEmail = normalizeEmail(email);
        return userAccountRepository.findByEmailIgnoreCase(normalizedEmail)
            .map(user -> attachPasswordToExistingUser(user, displayName, password))
            .orElseGet(() -> userAccountRepository.save(
                new UserAccount(normalizedEmail, displayName, passwordEncoder.encode(password))
            ));
    }

    @Transactional(readOnly = true)
    public UserAccount loginWithPassword(String email, String password) {
        UserAccount user = userAccountRepository.findByEmailIgnoreCase(normalizeEmail(email))
            .orElseThrow(() -> invalidCredentials());

        if (!user.hasPassword() || !passwordEncoder.matches(password, user.getPasswordHash())) {
            throw invalidCredentials();
        }

        return user;
    }

    @Transactional(readOnly = true)
    public UserAccount get(UUID userId) {
        return userAccountRepository.findById(userId)
            .orElseThrow(() -> new EntityNotFoundException("User not found: " + userId));
    }

    private UserAccount attachPasswordToExistingUser(UserAccount user, String displayName, String password) {
        if (user.hasPassword()) {
            throw new ResponseStatusException(HttpStatus.CONFLICT, "Email already has a password account");
        }

        user.updateDisplayName(displayName);
        user.changePasswordHash(passwordEncoder.encode(password));
        return user;
    }

    private ResponseStatusException invalidCredentials() {
        return new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Invalid email or password");
    }

    private String normalizeEmail(String email) {
        return email.trim().toLowerCase(Locale.ROOT);
    }
}
